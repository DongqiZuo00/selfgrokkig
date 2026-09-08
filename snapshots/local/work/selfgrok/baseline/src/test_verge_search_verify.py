"""Full-budget synthetic receipts only, never pretrained model outputs."""
import copy
import tempfile
from pathlib import Path
import unittest
from common import atomic_json,read_json,read_jsonl,group_advantages,stable_int
from verge_repair_sampling import atomic_jsonl
from verge_search_protocol import round_config,worker_config,PLAN
from verge_search_worker_protocol import worker_prepared
from verge_search_events import first_training_event,final_event,BUDGET
from verge_search_verify import validate_segment
from verge_independent_sampling import PROTOCOL
from verge_search_sampling_contract import training_request_seed,endpoint_request_seed
import json
from test_verge_search_protocol import EXP


class ReceiptTests(unittest.TestCase):
    def fixture(self,directory,has_event=False):
        cfg=worker_config(round_config(EXP,0,json.loads((EXP/PLAN).read_text())),2)
        frozen=worker_prepared(cfg,{'checkpoint':cfg['solver_start']})
        state={'train_tokens':0,'generated_tokens':0,'target_successes':0,'first_target_success':None,
            'nonzero_advantage_tokens':0,'optimizer_steps':0,'rollouts':0,
            'used_by_stage':[BUDGET],'group_by_stage':[32],'target_loss_tokens':BUDGET,'update':32,'masked_phase_boundary_groups':0}
        for update in range(1,33):
            seed=training_request_seed(cfg,update-1)
            rows=[{'instance_id':f'train{update}','instance_index':0,'rollout_index':i,'prompt_role':'target',
                'reward_group':update-1,'reward':int(has_event and update==1 and i==0),'prompt_token_ids':[1],
                'completion_token_ids':[2]*2048,'completion_tokens':2048,'max_tokens':2048,
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':seed,'sampling_child_seed':seed+i}
                for i in range(8)]
            masks=[[1]*2048 for _ in rows];advantages=group_advantages([r['reward'] for r in rows],8)
            state['first_target_success']=first_training_event(state,rows,masks,update)
            state['target_successes']+=sum(r['reward'] for r in rows)
            state['train_tokens']+=16384;state['generated_tokens']+=16384;state['rollouts']+=8
            state['nonzero_advantage_tokens']+=sum(2048 for a in advantages if abs(a)>1e-12)
            state['optimizer_steps']+=int(has_event)
            atomic_jsonl(directory/'train'/f'update_{update:04d}.jsonl',rows)
            atomic_json(directory/'train'/f'update_{update:04d}_allocation.json',{
                'stage':0,'policy_checkpoint_update':update-1,'advantage_source':'binary_complete_target_reward',
                'advantages':advantages,'loss_masks':masks})
            atomic_json(directory/'train'/f'update_{update:04d}_summary.json',{'optimizer_step':has_event})
        for kind,prompts,n in (('target',64,8),('scope',32,4)):
            seed=endpoint_request_seed(cfg,kind)
            request={'prompt':[[1]]*prompts,'n':n,'seed':seed,'max_tokens':2048,'temperature':1.,'top_p':1.,'top_k':-1}
            records=[{'instance_index':i//n,'rollout_index':i%n,'prompt_token_ids':[1],
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':seed+i//n*n,
                'sampling_child_seed':seed+i,'reward':0,'completion_token_ids':[2],
                'completion_tokens':1,'max_tokens':2048} for i in range(prompts*n)]
            response={'backend_protocol':PROTOCOL,'prompt_parent_seeds':[seed+i*n for i in range(prompts)],
                'child_seed_count':prompts*n,'choices':[{'index':i,'token_ids':[2]} for i in range(prompts*n)]}
            atomic_jsonl(directory/f'{kind}.jsonl',records)
            atomic_json(directory/f'{kind}.response.json',{'request':request,'response':response,'backend_protocol':PROTOCOL})
        result={'training':state,'checkpoint':f"checkpoints/{cfg['protocol_version']}_direct/resume_u0032",
            'fresh_optimizer':True,'weight_decay':0.,'kl_reference':cfg['solver_start'],'binary_rewards_only':True,
            'target':{'rollouts':512,'counts':[0]*11},'target_only_search':True,
            'search_comparison_complete':False,'search_round':cfg['search_round'],'search_branch_index':cfg['search_branch_index'],
            'training_random_stream_id':cfg['training_random_stream_id'],'endpoint_random_stream_id':cfg['random_stream_id']}
        result['search_event']=final_event(cfg,state,result['target'])
        return frozen,result

    def test_zero_rewards_complete_but_remain_censored(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p)
            self.assertTrue(validate_segment(f,r,p)['search_segment_verified'])
            self.assertTrue(r['search_event']['right_censored'])
            self.assertEqual(r['training']['optimizer_steps'],0)

    def test_training_event_not_pooled_into_endpoint_rate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p,True)
            validate_segment(f,r,p)
            self.assertEqual(r['search_event']['observation_loss_tokens'],16384)
            self.assertEqual(r['target']['counts'][-1],0)
            self.assertEqual(r['training']['optimizer_steps'],32)

    def test_event_and_policy_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p,True)
            for key,value in (('kl_reference','foreign'),('fresh_optimizer',False),('checkpoint','foreign')):
                changed=copy.deepcopy(r);changed[key]=value
                with self.assertRaises(ValueError):validate_segment(f,changed,p)
            r['search_event']['observation_loss_tokens']=BUDGET
            with self.assertRaises(ValueError):validate_segment(f,r,p)

    def test_modified_advantage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p)
            path=p/'train/update_0001_allocation.json';a=read_json(path);a['advantages'][0]=1;atomic_json(path,a)
            with self.assertRaises(ValueError):validate_segment(f,r,p)

    def test_another_training_branch_stream_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p);path=p/'train/update_0001.jsonl';rows=read_jsonl(path)
            seed=training_request_seed(dict(f['config'],training_random_stream_id='verge_search_v1_train_r00_b1'),0)
            for i,row in enumerate(rows):row.update(sampling_parent_seed=seed,sampling_child_seed=seed+i)
            atomic_jsonl(path,rows)
            with self.assertRaisesRegex(RuntimeError,'mapping'):validate_segment(f,r,p)

    def test_changed_endpoint_stream_and_worker_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p)
            changed=copy.deepcopy(r);changed['search_branch_index']=1
            with self.assertRaisesRegex(ValueError,'identity'):validate_segment(f,changed,p)
            path=p/'target.response.json';bundle=read_json(path);bundle['request']['seed']+=1;atomic_json(path,bundle)
            with self.assertRaisesRegex(ValueError,'Endpoint draw stream'):validate_segment(f,r,p)


if __name__=='__main__':unittest.main()
