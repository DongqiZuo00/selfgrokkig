"""Synthetic launch guards and shared-endpoint metadata; no model requests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from verge_search_protocol import PLAN,SUITE,round_config,worker_config,prepared_round
from verge_search_worker_protocol import worker_prepared,validate_prepared,validate_launch
from test_verge_search_protocol import EXP,endpoint


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class WorkerProtocolTests(unittest.TestCase):
    def fixture(self,exp):
        p=json.loads((EXP/PLAN).read_text())
        write(exp/'manifests/verge_mistral_round1.json',json.loads((EXP/'manifests/verge_mistral_round1.json').read_text()))
        cfg=round_config(exp,0,p);worker=worker_config(cfg,2)
        initial=endpoint(cfg['solver_start']);frozen=prepared_round(cfg,initial,.5)
        root=exp/'raw_results'/SUITE
        write(exp/PLAN,p);write(root/'protocol_frozen.json',p)
        write(exp/'manifests'/(cfg['protocol_version']+'.json'),cfg)
        write(root/'rounds'/cfg['protocol_version']/'round_frozen.json',frozen)
        for suite,file,flag,count,n in (
            ('verge_book_v2','PRIMARY_BLOCK_COMPLETE.json','primary_block_complete','outer_rounds',24),
            ('verge_control_v1','DIRECT_CONTROL_BLOCK_COMPLETE.json','control_block_complete','segments',18),
            ('verge_followon_v1','FOLLOWON_BLOCK_COMPLETE.json','followon_block_complete','candidates',72)):
            write(exp/'raw_results'/suite/file,{flag:True,count:n,'suite_complete':False})
        source=exp/cfg['solver_start'];write(source/'verge_committed.json',{'update':0})
        for file in ('adapter_model.safetensors','state.pt'):(source/file).write_bytes(b'synthetic-not-model-weights')
        return worker,frozen

    def test_launch_after_exact_frozen_predecessors(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);cfg,frozen=self.fixture(exp)
            self.assertEqual(validate_launch(exp,cfg),frozen)

    def test_metadata_copy_has_no_new_draws_and_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,shared=self.fixture(Path(tmp));f=worker_prepared(cfg,shared['initial'])
            self.assertEqual(f['additional_initial_model_draws'],0)
            self.assertEqual(len(f['branches']),1)
            self.assertEqual(f['branches'][0]['index'],0)
            validate_prepared(cfg,f)
            f['initial']['target']['counts'][0]=1
            self.assertEqual(shared['initial']['target']['counts'][0],0)
            f['branches'][0]['stages'][0]['tokens']-=1
            with self.assertRaises(RuntimeError):validate_prepared(cfg,f)

    def test_changed_training_or_endpoint_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);cfg,_=self.fixture(exp)
            for key,value in [('training_random_stream_id','another_branch'),('random_stream_id','another_round'),
                              ('seed',43),('book_suite','verge_book_v2'),('solver_start','checkpoints/foreign')]:
                changed=copy.deepcopy(cfg);changed[key]=value
                with self.assertRaises((RuntimeError,ValueError)):validate_launch(exp,changed)

    def test_incomplete_predecessor_and_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);cfg,_=self.fixture(exp)
            path=exp/'raw_results/verge_followon_v1/FOLLOWON_BLOCK_COMPLETE.json'
            original=json.loads(path.read_text());write(path,dict(original,candidates=71))
            with self.assertRaisesRegex(RuntimeError,'incomplete'):validate_launch(exp,cfg)
            write(path,original)
            plan=json.loads((exp/PLAN).read_text());plan['budget_note']='changed after submission'
            write(exp/'raw_results'/SUITE/'protocol_frozen.json',plan)
            with self.assertRaisesRegex(RuntimeError,'submitted plan'):validate_launch(exp,cfg)

    def test_changed_shared_draw_or_uncommitted_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);cfg,shared=self.fixture(exp)
            path=exp/'raw_results'/SUITE/'rounds'/cfg['search_round_version']/'round_frozen.json'
            changed=copy.deepcopy(shared);changed['branches'][0]['tokens']-=1;write(path,changed)
            with self.assertRaisesRegex(RuntimeError,'Shared initial'):validate_launch(exp,cfg)
            write(path,shared);write(exp/cfg['solver_start']/'verge_committed.json',{'update':1})
            with self.assertRaisesRegex(RuntimeError,'not committed'):validate_launch(exp,cfg)


if __name__=='__main__':unittest.main()
