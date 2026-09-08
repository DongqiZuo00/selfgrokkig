"""Full-budget synthetic receipts only, never pretrained model outputs."""
import copy
import tempfile
from pathlib import Path
import unittest
from common import atomic_json,read_json,group_advantages
from verge_repair_sampling import atomic_jsonl
from verge_followon_protocol import candidate_config,prepared_round
from verge_followon_events import first_training_event,final_event,BUDGET
from verge_followon_verify import validate_segment
from verge_independent_sampling import PROTOCOL
from test_verge_followon_protocol import bank,plan,EXP


class ReceiptTests(unittest.TestCase):
    def fixture(self,directory,has_event=False):
        cfg=candidate_config(EXP,bank(),0,plan())
        frozen=prepared_round(cfg,{'checkpoint':cfg['solver_start']})
        state={'train_tokens':0,'generated_tokens':0,'target_successes':0,'first_target_success':None,
            'nonzero_advantage_tokens':0,'optimizer_steps':0,'rollouts':0,
            'used_by_stage':[BUDGET],'group_by_stage':[16],'target_loss_tokens':BUDGET,'update':16}
        for update in range(1,17):
            rows=[{'instance_id':f'train{update}','instance_index':0,'rollout_index':i,'prompt_role':'target',
                'reward_group':update-1,'reward':int(has_event and update==1 and i==0),'prompt_token_ids':[1],
                'completion_token_ids':[2]*2048,'completion_tokens':2048,'max_tokens':2048,
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':update*8,'sampling_child_seed':update*8+i}
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
        for kind,prompts,n in (('target',64,8),('scope',32,4),('completion',64,1)):
            request={'prompt':[[1]]*prompts,'n':n,'seed':1000}
            records=[{'instance_index':i//n,'rollout_index':i%n,'prompt_token_ids':[1],
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':1000+i//n*n,
                'sampling_child_seed':1000+i,'reward':0} for i in range(prompts*n)]
            response={'backend_protocol':PROTOCOL,'prompt_parent_seeds':[1000+i*n for i in range(prompts)],
                'child_seed_count':prompts*n,'choices':[{'index':i} for i in range(prompts*n)]}
            atomic_jsonl(directory/f'{kind}.jsonl',records)
            atomic_json(directory/f'{kind}.response.json',{'request':request,'response':response,'backend_protocol':PROTOCOL})
        result={'training':state,'checkpoint':f"checkpoints/{cfg['protocol_version']}_direct/resume_u0016",
            'fresh_optimizer':True,'weight_decay':0.,'kl_reference':cfg['solver_start'],'binary_rewards_only':True,
            'target':{'rollouts':512,'counts':[0]*11},'completion':{'rollouts':64,'counts':[0]*11}}
        result['followon_event']=final_event(cfg['followon_source'],state,result['target'],result['completion'])
        return frozen,result

    def test_zero_rewards_complete_but_remain_censored(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p)
            self.assertTrue(validate_segment(f,r,p)['followon_segment_verified'])
            self.assertTrue(r['followon_event']['right_censored'])
            self.assertEqual(r['training']['optimizer_steps'],0)

    def test_training_event_not_pooled_into_endpoint_rate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p,True)
            validate_segment(f,r,p)
            self.assertEqual(r['followon_event']['observation_loss_tokens'],16384)
            self.assertEqual(r['target']['counts'][-1],0)
            self.assertEqual(r['training']['optimizer_steps'],16)

    def test_event_and_policy_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p,True)
            for key,value in (('kl_reference','foreign'),('fresh_optimizer',False),('checkpoint','foreign')):
                changed=copy.deepcopy(r);changed[key]=value
                with self.assertRaises(ValueError):validate_segment(f,changed,p)
            r['followon_event']['observation_loss_tokens']=BUDGET
            with self.assertRaises(ValueError):validate_segment(f,r,p)

    def test_modified_advantage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f,r=self.fixture(p)
            path=p/'train/update_0001_allocation.json';a=read_json(path);a['advantages'][0]=1;atomic_json(path,a)
            with self.assertRaises(ValueError):validate_segment(f,r,p)


if __name__=='__main__':unittest.main()
