"""Execute the real runner body with a tiny CPU model and synthetic transport.

Tests optimizer continuity and durable recovery without a pretrained model, GPU,
Slurm job, task audit or sampled experimental output.
"""
import ast
import copy
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
import torch
from common import atomic_json, read_json, stable_int
from verge_control_protocol import segment_config, validate_control_config
from verge_control_block import prepared_round, validate_prepared
from verge_control_training import prepare_group
from verge_repair_training import phase_boundary_masks, train_step
from verge_repair_protocol import prompt_role
from verge_repair_sampling import atomic_jsonl
from verge_independent_sampling import PROTOCOL

SOURCE = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4,4))

    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.weight[input_ids])


class ControlRunnerTests(unittest.TestCase):
    def execute(self, exp, *, mode="dense_then_binary", interrupt=False):
        phases = ([{"kind":"target","reward_mode":"dense","tokens":8},
                   {"kind":"target","reward_mode":"binary","tokens":8}]
                  if mode == "dense_then_binary" else [{"kind":"target","reward_mode":"binary","tokens":8}])
        cfg = segment_config(SOURCE,mode,0,phases=phases,
            starting_checkpoint="checkpoints/verge_book_v2_initial_solver/resume_u0000",
            block_protocol="manifests/verge_control_v1_direct_protocol.json")
        name = cfg["protocol_version"]
        root = exp / "raw_results" / name
        initial = {"checkpoint":cfg["solver_start"],"scope":{"full_pass_rate":0.,"mean_case_fraction":0.}}
        atomic_json(root / "round_frozen.json",prepared_round(cfg,initial))
        saved_paths, called_groups, interruption = {}, [], {"pending":interrupt}

        class Pool:
            def __init__(self,**kwargs): pass
            def __enter__(self): return ["synthetic://no-network"]
            def __exit__(self,*args): pass

        class Client:
            def __init__(self,*args): pass
            def use_base(self): pass
            def score_rows(self,rows,n,seed,path,**kwargs):
                called_groups.append(str(path))
                return [{"instance_id":rows[0]["id"],"instance_index":0,"rollout_index":i,
                    "reward":0,"passed_cases":i%4,"total_cases":4,"completion":str(i),
                    "prompt_token_ids":[0],"completion_token_ids":[1+i%3],"completion_tokens":1,
                    "sampling_protocol":PROTOCOL,"sampling_parent_seed":seed,"sampling_child_seed":seed+i,
                    "max_tokens":2048} for i in range(n)]

        def build(seed,source):
            model = TinyModel()
            if (source / "state.pt").is_file():
                model.load_state_dict(torch.load(source / "state.pt",weights_only=False)["tiny_model"])
            return object(),model

        def optimizer_for(model,lr):
            opt = torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=0.)
            return opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)

        def commit(model,opt,scheduler,branch,update,state):
            path = exp / "checkpoints" / branch / f"resume_u{update:04d}"
            path.mkdir(parents=True,exist_ok=True)
            torch.save({"tiny_model":model.state_dict(),"optimizer":opt.state_dict(),
                        "scheduler":scheduler.state_dict(),"runtime_state":copy.deepcopy(state)},path / "state.pt")
            atomic_json(path / "verge_committed.json",{"update":update})
            saved_paths[branch] = path
            return path

        def sync(model,client,branch,update):
            if update == 1 and interruption["pending"]:
                interruption["pending"] = False
                raise RuntimeError("Synthetic interruption AFTER durable commit")

        def endpoint(client,directory,*,checkpoint,selection,scope):
            for kind,prompts,n in (("target",64,8),("scope",32,4)):
                request = {"prompt":[[0]]*prompts,"n":n,"seed":1000}
                records = [{"instance_index":i//n,"rollout_index":i%n,"prompt_token_ids":[0],
                    "sampling_protocol":PROTOCOL,"sampling_parent_seed":1000+(i//n)*n,
                    "sampling_child_seed":1000+i} for i in range(prompts*n)]
                response = {"backend_protocol":PROTOCOL,"prompt_parent_seeds":[1000+i*n for i in range(prompts)],
                    "child_seed_count":prompts*n,"choices":[{"index":i} for i in range(prompts*n)]}
                atomic_json(directory / f"{kind}.response.json",{"backend_protocol":PROTOCOL,"request":request,"response":response})
                atomic_jsonl(directory / f"{kind}.jsonl",records)
            return {"checkpoint":checkpoint,"scope":{"full_pass_rate":0.,"mean_case_fraction":0.}}

        # Extract the actual source function; only dependencies are injected.
        tree = ast.parse((SOURCE / "src/verge_control_train.py").read_text())
        function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "main")
        code = compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),"verge_control_train.py","exec")
        env = dict(globals(),EXP_ROOT=exp,ROOT=root,VERSION=name,config=lambda:cfg,
            rows=lambda path:[{"id":"synthetic"}], cycle_rows=lambda rows,*args:rows[:1],
            VLLMServerPool=Pool,VLLMRolloutClient=Client,build_recipe_model=build,
            build_reference_model=lambda *args:TinyModel(),optimizer_for=optimizer_for,
            commit=commit,latest=lambda branch:saved_paths.get(branch),sync_adapter=sync,endpoint=endpoint)
        exec(code,env)
        with redirect_stdout(io.StringIO()):
            if interrupt:
                with self.assertRaisesRegex(RuntimeError,"Synthetic interruption"):
                    env["main"]()
            env["main"]()
            group_count = len(called_groups)
            env["main"]()  # Completed recovery must not generate another group.
            self.assertEqual(len(called_groups),group_count)
        result = read_json(root / "branches/0/complete.json")
        tensor = torch.load(exp / result["checkpoint"] / "state.pt",weights_only=False)["tiny_model"]["weight"]
        return result,tensor,called_groups

    def test_dense_then_binary_keeps_optimizer_history_but_raw_events_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            result,tensor,groups = self.execute(Path(temp))
            self.assertEqual(result["training"]["used_by_stage"],[8,8])
            self.assertEqual(result["training"]["optimizer_steps"],2)
            self.assertEqual(result["training"]["target_successes"],0)
            self.assertIsNone(result["training"]["first_target_success"])
            self.assertTrue(torch.count_nonzero(tensor))
            self.assertEqual(len(groups),2)
            self.assertFalse(result["control_comparison_complete"])

    def test_post_commit_recovery_matches_uninterrupted_tiny_model(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            plain,weight_a,groups_a = self.execute(Path(first))
            resumed,weight_b,groups_b = self.execute(Path(second),interrupt=True)
            self.assertTrue(torch.equal(weight_a,weight_b))
            self.assertEqual(plain["training"]["train_tokens"],resumed["training"]["train_tokens"])
            self.assertEqual(len(groups_a),len(groups_b))

    def test_fresh_all_zero_binary_branch_has_no_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temp:
            result,tensor,groups = self.execute(Path(temp),mode="binary")
            self.assertEqual(result["training"]["optimizer_steps"],0)
            self.assertEqual(result["training"]["nonzero_advantage_tokens"],0)
            self.assertFalse(torch.count_nonzero(tensor))
            self.assertEqual(len(groups),1)


if __name__ == "__main__":
    unittest.main()
