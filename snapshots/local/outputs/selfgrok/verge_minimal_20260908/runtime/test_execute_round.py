"""CPU interface fixtures only: no model, Torch import, GPU, or real training."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "round/fixtures")]
from fixture_round import request as fixture_request
import execute_round as ex
from round_cli import build_plan, canonical, sha, write_json


class FixtureSolver:
    """Explicit fake tensor/model backend used only by these CPU tests."""
    def __init__(self, adapter, device):
        self.adapter = Path(adapter)
        self.value = int((self.adapter / "adapter_model.safetensors").read_text())
        self.optimizer = SimpleNamespace(state={}, param_groups=[{"weight_decay": 0}])
        self.optimizer_steps = 0
        self.generated_tokens = 0
        self.calls = []
        self.saved = []
        self.closed = False
        self.create_factory = lambda program: SimpleNamespace(nodes={
            "start": SimpleNamespace(routes=[SimpleNamespace(target="end")]),
            "end": SimpleNamespace(routes=[])})

    def generate(self, row, n, cap, seed, *, output=None, target=False):
        self.calls.append({"n": n, "cap": cap, "seed": seed, "target": target, "row": row["id"]})
        token_lists = [[7] * min(cap, 2) for _ in range(n)]
        rewards = [0 if target and self.value == 0 else i % 2 for i in range(n)]
        text = "START start:\n    NEXT end\nEND end"
        verification = [{"reward": reward, "parse_valid": True,
            "per_test": [{"input": case["input"], "pass": reward} for case in row["ground_truth"]]} for reward in rewards]
        raw = {"mode": "SYNTHETIC_UNIT_FIXTURE", "decoding_policy": "dsl_grammar_v1",
               "rewards": rewards, "prompt_token_ids": [1], "completion_token_ids": token_lists,
               "verifier_completions": [text] * n, "verification": verification}
        self.generated_tokens += sum(map(len, token_lists))
        if output:
            write_json(output, raw)
        return raw

    def update(self, raw):
        if len(set(raw["rewards"])) != 2:
            raise AssertionError("fixture optimizer must never see a constant reward group")
        self.optimizer_steps += 1
        self.value += 1
        return {"optimizer_step": True, "parameters_changed": True, "optimizer_steps": self.optimizer_steps}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_text(str(self.value))
        self.saved.append(str(path))
        return str(path)

    def close(self):
        self.closed = True


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base_path = self.root / "source_base"
        self.base_path.mkdir()
        (self.base_path / "adapter_model.safetensors").write_text("0")
        req = fixture_request()
        req["base_checkpoint"] = str(self.base_path)
        req["proposal"]["base_checkpoint"] = str(self.base_path)
        req["challenger"]["raw_output"] = canonical(req["proposal"])
        req["selection"]["prompt_view_sha256"] = sha("SYNTHETIC_UNIT_VIEW")
        self.plan = build_plan(req, _fixture=True)
        cases = [{"test_id": t, "input": t, "expected_output": t, "expected_accepted": True,
                  "check_output": True} for t in ("t1", "t2", "t3", "t4")]
        messages = [{"role": "user", "content": "SYNTHETIC UNIT prompt"}]
        self.rows = {"manifest_id": self.plan["selection"]["manifest_id"],
                     "prompt_view_sha256": self.plan["selection"]["prompt_view_sha256"],
                     "train": [{"id": "train-1", "messages": messages, "ground_truth": cases, "hint": None}],
                     "selection": [{"id": i, "messages": messages, "ground_truth": cases, "hint": None}
                                   for i in self.plan["selection"]["instance_ids"]]}
        self.catalogue = {s["stage_id"]: {**s, "rows": self.rows["train"]}
                          for s in req["proposal"]["curricula"]["g1"]}
        self.created = []
        def factory(adapter, device):
            solver = FixtureSolver(adapter, device)
            self.created.append(solver)
            return solver
        self.factory = factory
        self.fingerprint = lambda solver: sha(solver.value)

    def base(self, preproposal=False):
        plan = ex.build_baseline_plan(base_checkpoint=str(self.base_path), training_seed=self.plan["training_seed"],
                                      selection=self.plan["selection"]) if preproposal else self.plan
        return ex.execute_base(plan, self.rows, self.root / "base_output", solver_factory=self.factory,
                               fingerprint=self.fingerprint, _fixture=True)

    def branch(self, name, base):
        return ex.execute_branch(self.plan, self.rows, self.catalogue, base, name, self.root / name,
                                  solver_factory=self.factory, fingerprint=self.fingerprint, _fixture=True)

    def test_base_only_batches_are_8_and_seeds_are_chunk_not_per_sample(self):
        base = self.base()
        solver = self.created[0]
        self.assertEqual(len(solver.calls), 32)
        self.assertTrue(all(call["n"] == 8 and call["cap"] == 2048 for call in solver.calls))
        records = base["evaluations"]["base"]["records"]
        self.assertEqual(len(records), 256)
        self.assertEqual(len({r["chunk_seed"] for r in records[:8]}), 1)
        self.assertEqual({r["sample_index_in_chunk"] for r in records[:8]}, set(range(8)))
        self.assertTrue(all("global_rollout_index" not in r for r in records))
        self.assertEqual(solver.optimizer_steps, 0)
        self.assertEqual(solver.saved, [])
        self.assertTrue(solver.closed)

    def test_preproposal_base_contract_reuses_without_fabricated_q_plan(self):
        base = self.base(preproposal=True)
        self.assertIsNone(base["plan_sha256"])
        self.assertTrue(base["preproposal_base"])
        self.assertEqual(base["baseline_contract_sha256"], sha(ex.baseline_contract(self.plan)))
        direct = self.branch("direct", base)
        self.assertEqual(direct["execution"]["actual_optimizer_steps"], 0)
        changed = copy.deepcopy(self.plan)
        changed["training_seed"] += 1
        with self.assertRaisesRegex(ValueError, "base sampling contract"):
            ex.validate_base_fragment(changed, base)

    def test_direct_constant_groups_never_update_save_or_resample_target_eval(self):
        base = self.base()
        direct = self.branch("direct", base)
        solver = self.created[-1]
        self.assertEqual(solver.optimizer_steps, 0)
        self.assertEqual(solver.saved, [])
        self.assertEqual(len(solver.calls), 4)  # four training groups only; eval aliases hit actual base cache
        self.assertTrue(all(e["evaluation_cache_hit"] for e in direct["evaluations"].values()))
        self.assertTrue(all(c["path"] == str(self.base_path) for c in direct["checkpoints"].values()))
        self.assertEqual(direct["execution"]["training_generated_tokens"], 32)

    def test_mixed_training_uses_backend_update_and_saves_changed_weights_only(self):
        base = self.base()
        branch = self.branch("g1", base)
        solver = self.created[-1]
        self.assertGreater(solver.optimizer_steps, 0)
        self.assertEqual(branch["execution"]["actual_optimizer_steps"], solver.optimizer_steps)
        self.assertEqual(len(branch["branches"]["g1"]["actual_model_updates"]), solver.optimizer_steps)
        self.assertEqual((self.base_path / "adapter_model.safetensors").read_text(), "0")
        self.assertTrue(all(Path(p).is_relative_to(self.root / "g1") for p in solver.saved))

    def test_all_branch_fragments_merge_and_score_using_same_interface(self):
        base = self.base(preproposal=True)
        branches = [self.branch(name, base) for name in ("direct", "g1", "g2", "g3")]
        result = ex.merge_fragments(self.plan, base, branches, self.root / "merged", _fixture=True)
        self.assertEqual(result["summary"]["status"], "complete")
        self.assertEqual(result["summary"]["total_training_generated_tokens"], 128)
        self.assertTrue(result["summary"]["at_least_one_curriculum_mixed_update"])
        self.assertGreater(result["summary"]["cached_evaluation_aliases"], 0)
        self.assertLess(result["summary"]["evaluation_rollouts"], result["summary"]["logical_evaluation_rollout_slots"])
        self.assertFalse(result["events"]["first_success_order_observable"])

    def test_graph_cycle_uses_actual_nodes_routes_including_unreachable_component(self):
        def node(*targets):
            return SimpleNamespace(routes=[SimpleNamespace(target=t) for t in targets])
        factory = SimpleNamespace(nodes={"start": node("end"), "end": node(), "unused": node("unused")})
        self.assertTrue(ex.graph_has_cycle(factory))
        factory.nodes["unused"] = node("NONE")
        self.assertFalse(ex.graph_has_cycle(factory))

    def test_default_factory_explicitly_enables_grammar(self):
        calls = []
        with patch.dict(sys.modules, {"hf_backend": SimpleNamespace(Solver=lambda **kw: calls.append(kw) or object())}):
            ex._solver_factory("adapter", 2)
        self.assertEqual(calls, [{"adapter": "adapter", "device": 2, "grammar_enabled": True}])

    def test_frozen_input_interface_rejects_hint_and_wrong_plan(self):
        path = self.root / "target_rows.json"
        rows = copy.deepcopy(self.rows)
        rows["selection"][0]["hint"] = "forbidden"
        write_json(path, rows)
        with self.assertRaisesRegex(ValueError, "unhinted"):
            ex.load_inputs(self.plan, path, _fixture=True)
        wrong = copy.deepcopy(self.plan)
        wrong["B"] = 1
        with self.assertRaisesRegex(ValueError, "changed after freezing"):
            ex.checked_plan(wrong)


if __name__ == "__main__":
    unittest.main(verbosity=2)
