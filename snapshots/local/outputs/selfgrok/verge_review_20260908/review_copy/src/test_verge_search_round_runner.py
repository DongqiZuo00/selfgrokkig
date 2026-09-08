"""Execute real prepare/finish functions with injected synthetic transport.

No pretrained model, Slurm or network request. The four worker outcomes are
synthetic; their training receipts have separate full-budget tests.
"""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
from common import atomic_json,read_json,read_jsonl,stable_int
from verge_search_protocol import prepared_round,select_round,version
from verge_search_worker_protocol import validate_round_launch
from verge_search_round_verify import validate_initial,validate_completion
from verge_search_sampling_contract import endpoint_request_seed
import verge_search_controller as ctl
import test_verge_search_worker_protocol as worker_tests
from test_verge_search_round_verify import batch

SOURCE=Path(__file__).resolve().parent/'verge_search_round.py'


class RoundRunnerTests(unittest.TestCase):
    def harness(self,exp,*,interrupt=False,missing=False):
        worker_tests.WorkerProtocolTests().fixture(exp)
        cfg=read_json(exp/'manifests/verge_search_v1_r00.json');name=cfg['protocol_version']
        root=exp/'raw_results/verge_search_v1/rounds'/name
        (root/'round_frozen.json').unlink()  # Synthetic fixture only; real results are untouched.
        calls={'initial':0,'completion_requests':0,'new_completion_batches':0,'loaded':[],'pools':0}
        state={'interrupt':interrupt}

        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):calls['pools']+=1;return ['synthetic://no-network']
            def __exit__(self,*args):pass

        class Client:
            def __init__(self,*args):pass
            def load_lora(self,label,checkpoint):calls['loaded'].append((label,str(checkpoint.relative_to(exp))))
            def use_base(self):pass
            def score_rows(self,rows,n,seed,path,**kwargs):
                calls['completion_requests']+=1
                if (len(rows),n,path.name)!=(64,1,'completion.jsonl'):raise AssertionError('Unexpected model draw')
                if stable_int(seed,kwargs['sampling_seed_key'])!=endpoint_request_seed(cfg,'completion'):
                    raise AssertionError('Changed completion stream')
                if not path.exists():
                    batch(path.parent,cfg,'completion',64,1);calls['new_completion_batches']+=1
                return read_jsonl(path)

        def endpoint(client,path,*,checkpoint,selection,scope):
            calls['initial']+=1
            target=batch(path,cfg,'target',64,8);batch(path,cfg,'scope',32,4)
            initial={'checkpoint':checkpoint,'target':target,'scope':{'full_pass_rate':0.,'mean_case_fraction':0.}}
            atomic_json(path/'endpoint.json',initial);return initial

        def outcomes(r,b,*,deep=False):
            if r!=0 or not deep:raise AssertionError('Missing deep worker validation request')
            if missing and b==3:return None
            initial=read_json(root/'round_frozen.json')['initial'];result=copy.deepcopy(initial)
            result.update(checkpoint=f'checkpoints/{version(0,b)}_direct/resume_u0032',
                training={'update':32,'optimizer_steps':32 if b==1 else 0,'train_tokens':524288,'target_successes':0})
            if b==1:
                result['target']['counts']=[512]*11;result['target']['rates']=[1.]*11
                result['target']['keyed_vectors']={k:[1]*11 for k in result['target']['keyed_vectors']}
            return result

        def profile(records,rows):
            return {'rollouts':64,'counts':[0]*11,'rates':[0.]*11,
                    'keyed_vectors':{f"{r['instance_id']}::{r['rollout_index']}":[0]*11 for r in records}}

        def write(path,value):
            atomic_json(path,value)
            if path.name=='completion_profile.json' and state['interrupt']:
                state['interrupt']=False;raise RuntimeError('Synthetic post-completion commit interruption')

        tree=ast.parse(SOURCE.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        code=compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(SOURCE),'exec')
        env=dict(globals(),EXP_ROOT=exp,ROOT=root,VERSION=name,config=lambda:cfg,VLLMServerPool=Pool,
            VLLMRolloutClient=Client,endpoint=endpoint,profile=profile,atomic_json=write,
            rows=lambda path:[{'id':str(i)} for i in range(64)])
        exec(code,env)
        stack=ExitStack();stack.enter_context(patch.object(ctl,'EXP',exp))
        stack.enter_context(patch.object(ctl,'ROOT',exp/'raw_results/verge_search_v1'))
        stack.enter_context(patch.object(ctl,'completed_worker',side_effect=outcomes))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return env,calls,root,stack

    def test_shared_prepare_is_idempotent_without_worker_initial_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,calls,root,stack=self.harness(Path(tmp))
            with stack:
                env['prepare']();before=read_json(root/'round_frozen.json');env['prepare']()
                self.assertEqual(read_json(root/'round_frozen.json'),before)
                self.assertEqual(calls['initial'],1);self.assertEqual(calls['completion_requests'],0)

    def test_only_selected_checkpoint_gets_completion_and_no_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,calls,root,stack=self.harness(Path(tmp))
            with stack:
                env['prepare']();env['finish']();decision=read_json(root/'decision.json');record=read_json(root/'complete.json')
                env['finish']()
                self.assertEqual(calls['new_completion_batches'],1);self.assertEqual(calls['completion_requests'],1)
                self.assertEqual(calls['loaded'][-1][1],decision['selected']['checkpoint'])
                self.assertEqual(record['selected'],decision['selected']);self.assertFalse(record['completion_used_for_selection'])
                self.assertTrue(record['search_round_complete']);self.assertFalse(record['search_block_complete'])
                self.assertFalse(record['suite_complete'])

    def test_post_completion_commit_resume_reuses_exact_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,calls,root,stack=self.harness(Path(tmp),interrupt=True)
            with stack:
                env['prepare']()
                with self.assertRaisesRegex(RuntimeError,'post-completion'):env['finish']()
                self.assertFalse((root/'complete.json').exists());decision=read_json(root/'decision.json')
                env['finish']();self.assertEqual(read_json(root/'complete.json')['selected'],decision['selected'])
                self.assertEqual(calls['completion_requests'],1);self.assertEqual(calls['new_completion_batches'],1)

    def test_missing_worker_cannot_sample_or_complete_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,calls,root,stack=self.harness(Path(tmp),missing=True)
            with stack:
                env['prepare']()
                with self.assertRaisesRegex(RuntimeError,'four'):env['finish']()
                self.assertEqual(calls['completion_requests'],0);self.assertFalse((root/'complete.json').exists())

    def test_changed_saved_decision_stops_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            env,calls,root,stack=self.harness(Path(tmp))
            with stack:
                env['prepare']();atomic_json(root/'decision.json',{'selected':'altered'})
                with self.assertRaisesRegex(RuntimeError,'changed'):env['finish']()
                self.assertEqual(calls['completion_requests'],0)


if __name__=='__main__':unittest.main()
