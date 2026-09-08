"""Actual runner function on a 16-parameter CPU model, not a real experiment.

The synthetic budget is 16 tokens only inside this injected test harness.
Production configuration remains 262144; no model weights, GPUs or Slurm used.
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
from unittest.mock import patch
import torch
from common import atomic_json,read_json,stable_int,group_advantages
from verge_repair_sampling import atomic_jsonl
from verge_repair_training import phase_boundary_masks,train_step
from verge_repair_protocol import prompt_role
from verge_followon_protocol import candidate_config,prepared_round,validate_prepared
from verge_followon_events import first_training_event,final_event
from verge_independent_sampling import PROTOCOL
from test_verge_followon_protocol import bank,plan,EXP


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(4,4))
    def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.weight[input_ids])


class RunnerTests(unittest.TestCase):
    def execute(self,exp,*,mixed=False,interrupt=False):
        cfg=candidate_config(EXP,bank(),0,plan())
        cfg.update(train_tokens_per_branch=16,token_update_threshold=8,
                   followon_phases=[{'kind':'target','reward_mode':'binary','tokens':16}])
        name=cfg['protocol_version'];root=exp/'raw_results'/name
        initial={'checkpoint':cfg['solver_start'],'scope':{'full_pass_rate':0.,'mean_case_fraction':0.}}
        atomic_json(root/'round_frozen.json',prepared_round(cfg,initial))
        saved_paths={};called=[];fault={'pending':interrupt}

        def evaluation(directory,kind,prompts,n):
            request={'prompt':[[0]]*prompts,'n':n,'seed':1000}
            records=[{'instance_index':i//n,'rollout_index':i%n,'prompt_token_ids':[0],
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':1000+i//n*n,
                'sampling_child_seed':1000+i,'reward':0} for i in range(prompts*n)]
            response={'backend_protocol':PROTOCOL,'prompt_parent_seeds':[1000+i*n for i in range(prompts)],
                'child_seed_count':prompts*n,'choices':[{'index':i} for i in range(prompts*n)]}
            atomic_json(directory/f'{kind}.response.json',{'request':request,'response':response,'backend_protocol':PROTOCOL})
            atomic_jsonl(directory/f'{kind}.jsonl',records)
            return records

        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return ['synthetic://no-network']
            def __exit__(self,*args):pass

        class Client:
            def __init__(self,*args):pass
            def use_base(self):pass
            def score_rows(self,rows,n,seed,path,**kwargs):
                if path.name=='completion.jsonl':return evaluation(path.parent,'completion',len(rows),n)
                called.append(str(path))
                return [{'instance_id':rows[0]['id'],'instance_index':0,'rollout_index':i,
                    'reward':int(mixed and i==0),'completion':str(i),
                    'prompt_token_ids':[0],'completion_token_ids':[1+i%3],'completion_tokens':1,
                    'sampling_protocol':PROTOCOL,'sampling_parent_seed':seed,'sampling_child_seed':seed+i,
                    'max_tokens':2048} for i in range(n)]

        def build(seed,source):
            model=TinyModel()
            if (source/'state.pt').exists():model.load_state_dict(torch.load(source/'state.pt',weights_only=False)['tiny_model'])
            return object(),model

        def optimizer_for(model,lr):
            opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=0.)
            return opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)

        def commit(model,opt,scheduler,branch,update,state):
            path=exp/'checkpoints'/branch/f'resume_u{update:04d}';path.mkdir(parents=True,exist_ok=True)
            torch.save({'tiny_model':model.state_dict(),'optimizer':opt.state_dict(),
                'scheduler':scheduler.state_dict(),'runtime_state':copy.deepcopy(state)},path/'state.pt')
            atomic_json(path/'verge_committed.json',{'update':update});saved_paths[branch]=path
            return path

        def sync(model,client,branch,update):
            if update==1 and fault['pending']:
                fault['pending']=False;raise RuntimeError('Synthetic post-commit interruption')

        def endpoint(client,directory,*,checkpoint,selection,scope):
            evaluation(directory,'target',64,8);evaluation(directory,'scope',32,4)
            return {'checkpoint':checkpoint,'scope':{'full_pass_rate':0.,'mean_case_fraction':0.},
                    'target':{'rollouts':512,'counts':[0]*11}}

        tree=ast.parse((EXP/'src/verge_followon_train.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        code=compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                     'verge_followon_train.py','exec')
        env=dict(globals(),EXP_ROOT=exp,ROOT=root,VERSION=name,config=lambda:cfg,
            VLLMServerPool=Pool,VLLMRolloutClient=Client,build_recipe_model=build,
            build_reference_model=lambda *args:TinyModel(),optimizer_for=optimizer_for,
            commit=commit,latest=lambda branch:saved_paths.get(branch),sync_adapter=sync,
            rows=lambda path:[{'id':str(i)} for i in range(64)],
            cycle_rows=lambda dataset,*args:[dataset[0]],endpoint=endpoint,
            profile=lambda records,rows:{'rollouts':len(records),'counts':[0]*11})
        exec(code,env)
        with patch('verge_followon_events.BUDGET',16),patch('verge_followon_verify.BUDGET',16),redirect_stdout(io.StringIO()):
            if interrupt:
                with self.assertRaisesRegex(RuntimeError,'post-commit'):env['main']()
            env['main']()
        result=read_json(root/'branches/0/complete.json')
        stored=torch.load(saved_paths[name+'_direct']/'state.pt',weights_only=False)
        return result,stored,called

    def test_all_zero_target_stalls_and_is_censored(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,stored,calls=self.execute(Path(tmp))
            self.assertEqual(result['training']['optimizer_steps'],0)
            self.assertTrue(result['followon_event']['right_censored'])
            self.assertEqual(result['training']['train_tokens'],16)
            self.assertEqual(len(calls),2)

    def test_real_runner_keeps_training_event_out_of_endpoint_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,stored,calls=self.execute(Path(tmp),mixed=True)
            self.assertEqual(result['training']['optimizer_steps'],2)
            self.assertEqual(result['followon_event']['observation_loss_tokens'],8)
            self.assertEqual(result['target']['counts'][-1],0)
            self.assertGreater(float(stored['tiny_model']['weight'].abs().sum()),0)

    def test_post_commit_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            expected,weights,_=self.execute(Path(a),mixed=True)
            actual,resumed,calls=self.execute(Path(b),mixed=True,interrupt=True)
            self.assertTrue(torch.equal(weights['tiny_model']['weight'],resumed['tiny_model']['weight']))
            self.assertEqual(actual['followon_event'],expected['followon_event'])
            self.assertEqual(len(calls),2)
            self.assertEqual(len(set(calls)),2)


if __name__=='__main__':unittest.main()
