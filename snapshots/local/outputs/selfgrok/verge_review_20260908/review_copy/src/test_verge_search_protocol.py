"""Synthetic target-only search contracts; no new training run or model draw."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import verge_search_protocol as search

EXP=Path(__file__).resolve().parents[1]


def endpoint(checkpoint,successes=0,steps=0,scope=1.):
    vectors={f'i{i:02d}::{j}':[int(8*i+j<successes)]*11 for i in range(64) for j in range(8)}
    return {'checkpoint':checkpoint,'target':{'rollouts':512,'counts':[successes]*11,
        'rates':[successes/512]*11,'keyed_vectors':vectors,'bottleneck_index':0},
        'scope':{'full_pass_rate':scope,'mean_case_fraction':scope},
        'training':{'update':32,'optimizer_steps':steps,'train_tokens':524288,'target_successes':0}}


class TargetOnlySearchTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((EXP/search.PLAN).read_text())
        self.cfg=search.round_config(EXP,0,self.plan)

    def frozen(self):return search.prepared_round(self.cfg,endpoint(search.ROOT_CHECKPOINT),.01)

    def branches(self):return {i:endpoint(f'checkpoints/{search.version(0,i)}_direct/resume_u0032') for i in range(4)}

    def test_fixed_scope_and_budget(self):
        self.assertEqual(self.plan['total_loss_tokens'],6*4*524288)
        for k,v in [('training_seed',43),('branches_per_round',3),('completion_cap',4096),('maximum_concurrent_workers',3),
                    ('scope_max_drop',.05),('target_reward','dense'),('total_compute_matched',True)]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.assertRaises(ValueError):search.validate_plan(p)

    def test_four_unique_training_streams_shared_endpoint_stream(self):
        workers=[search.worker_config(self.cfg,i) for i in range(4)]
        self.assertEqual(len({c['training_random_stream_id'] for c in workers}),4)
        self.assertEqual(len({c['random_stream_id'] for c in workers}),1)
        self.assertTrue(all(c['seed']==42 and c['proposal_count']==0 and c['solver_start']==search.ROOT_CHECKPOINT for c in workers))
        self.assertNotIn('book_suite',self.cfg)

    def test_selector_refuses_changed_optimizer_or_sampling_contract(self):
        for key,value in [('weight_decay',.01),('solver_start','foreign'),('book_suite','verge_book_v2'),
                          ('scope_max_drop',.1),('seed',43),('train_tokens_per_branch',1048576)]:
            f=self.frozen();f['config'][key]=value
            with self.assertRaises(ValueError):search.select_round(f,self.branches())

    def test_round_and_branch_bounds(self):
        for r in [-1,6,True]:
            with self.assertRaises(ValueError):search.version(r)
        for b in [-1,4,True]:
            with self.assertRaises(ValueError):search.worker_config(self.cfg,b)

    def test_exact_previous_selection_required(self):
        prior={'protocol_version':search.version(0),'search_round_complete':True,'suite_complete':False,
               'selected':{'checkpoint':f'checkpoints/{search.version(0,2)}_direct/resume_u0032'}}
        cfg=search.round_config(EXP,1,self.plan,prior);self.assertEqual(cfg['solver_start'],prior['selected']['checkpoint'])
        prior['selected']['checkpoint']='checkpoints/verge_book_v2_uncertainty_r02_candidate_1/resume_u0032'
        with self.assertRaises(ValueError):search.round_config(EXP,1,self.plan,prior)
        with self.assertRaises(ValueError):search.round_config(EXP,0,self.plan,prior)

    def test_retained_root_may_persist_but_future_checkpoint_rejected(self):
        prior={'protocol_version':search.version(3),'search_round_complete':True,'suite_complete':False,
               'selected':{'checkpoint':search.ROOT_CHECKPOINT}}
        self.assertEqual(search.round_config(EXP,4,self.plan,prior)['solver_start'],search.ROOT_CHECKPOINT)
        prior['selected']['checkpoint']=f'checkpoints/{search.version(5,0)}_direct/resume_u0032'
        with self.assertRaises(ValueError):search.round_config(EXP,4,self.plan,prior)

    def test_no_generated_proposals_and_immutable_selection_draw(self):
        f=self.frozen();self.assertFalse(f['generation']['model_generated']);self.assertEqual(len(f['branches']),4)
        with self.assertRaises(ValueError):search.prepared_round(self.cfg,endpoint('foreign'),.5)
        with self.assertRaises(ValueError):search.prepared_round(self.cfg,endpoint(search.ROOT_CHECKPOINT),1.)

    def test_all_zero_steps_identity_keeps_start_despite_sampling_difference(self):
        f=self.frozen();f['initial']=endpoint(search.ROOT_CHECKPOINT,12)
        b=self.branches();b[1]['target']=endpoint('unused',100)['target']
        d=search.select_round(f,b)
        self.assertEqual(d['selected']['name'],'start');self.assertEqual(d['selected']['gain'],0.)
        self.assertEqual(d['target_condition_gains'],[0.,0.,0.])
        self.assertEqual(f['initial']['target']['counts'][0],12)
        self.assertFalse(d['challenger_update_allowed']);self.assertFalse(d['search_round_complete'])

    def test_same_selector_positive_branch_and_scope_guard(self):
        b=self.branches();b[1]=endpoint(b[1]['checkpoint'],512,32);b[2]=endpoint(b[2]['checkpoint'],512,32,.2)
        d=search.select_round(self.frozen(),b)
        self.assertEqual(d['selected']['name'],'candidate_1');self.assertFalse(d['evaluated_candidates'][1]['eligible'])
        self.assertEqual(d['selection_rung_zero_based'],10);self.assertEqual(d['reward_rung_zero_based'],10)
        self.assertFalse(d['challenger_update_allowed'])

    def test_incomplete_foreign_or_overbudget_branch_rejected(self):
        for field,value in [('train_tokens',524287),('update',101),('optimizer_steps',33)]:
            b=self.branches();b[2]['training'][field]=value
            with self.assertRaises(ValueError):search.select_round(self.frozen(),b)
        b=self.branches();del b[3]
        with self.assertRaises(ValueError):search.select_round(self.frozen(),b)

    def test_trajectory_success_cannot_substitute_for_fixed_endpoint(self):
        b=self.branches();b[1]['training']['target_successes']=100
        d=search.select_round(self.frozen(),b)
        self.assertEqual(d['target_condition_gains'],[0.,0.,0.]);self.assertEqual(d['selected']['name'],'start')


if __name__=='__main__':unittest.main()
