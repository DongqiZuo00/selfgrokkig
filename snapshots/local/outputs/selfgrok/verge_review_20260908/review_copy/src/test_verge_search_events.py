"""Pure event-clock tests. No pretrained sampling or eligibility audit."""
import copy
import unittest
from verge_search_events import BUDGET,first_training_event,final_event


def state():
    return {'train_tokens':0,'generated_tokens':0,'first_target_success':None,'target_successes':0}


def group(success=False):
    return [{'prompt_role':'target','instance_id':'synthetic','rollout_index':i,'reward_group':0,
             'reward':int(success and i==3),'completion_tokens':2} for i in range(8)]


class EventTests(unittest.TestCase):
    def test_event_clock_is_group_end_not_full_search_compute(self):
        rows=group(True);s=state();event=first_training_event(s,rows,[[1,1]]*8,1)
        self.assertEqual(event['observation_loss_tokens'],16)
        self.assertEqual(event['role'],'search_target_training')
        self.assertIsNone(s['first_target_success'])
        s.update(train_tokens=BUDGET,first_target_success=event,target_successes=1)
        record=final_event({'search_round':0,'search_branch_index':2},s,{'rollouts':512,'counts':[0]*11})
        self.assertEqual(record['target_endpoint_full_successes'],0)
        self.assertFalse(record['clock_is_total_search_compute'])
        self.assertTrue(record['selected_round_completion_pending'])

    def test_zero_target_right_censored_and_endpoint_event_is_separate(self):
        s=state();s['train_tokens']=BUDGET;cfg={'search_round':2,'search_branch_index':1}
        target={'rollouts':512,'counts':[0]*11}
        self.assertTrue(final_event(cfg,s,target)['right_censored'])
        target['counts']=[1]*11;r=final_event(cfg,s,target)
        self.assertFalse(r['right_censored']);self.assertEqual(r['observation_loss_tokens'],BUDGET)
        self.assertEqual(r['first_event']['role'],'search_target_selection')
        self.assertEqual(r['target_training_full_successes'],0)

    def test_incomplete_budget_or_disagreeing_event_is_not_completed_censor(self):
        s=state();cfg={'search_round':0,'search_branch_index':0};target={'rollouts':512,'counts':[0]*11}
        with self.assertRaises(ValueError):final_event(cfg,s,target)
        s.update(train_tokens=BUDGET,target_successes=1)
        with self.assertRaises(ValueError):final_event(cfg,s,target)

    def test_nonbinary_incomplete_or_non_target_group_rejected(self):
        original=group()
        for key,value in [('reward',.5),('prompt_role','scope'),('completion_tokens',2049),('rollout_index',7)]:
            rows=copy.deepcopy(original);rows[0][key]=value
            with self.assertRaises(ValueError):first_training_event(state(),rows,[[1,1]]*8,1)
        with self.assertRaises(ValueError):first_training_event(state(),original[:7],[[1,1]]*7,1)


if __name__=='__main__':unittest.main()
