"""Synthetic grouped event clocks, censoring and source-isolation checks."""
import copy
import unittest
from verge_followon_events import BUDGET,first_training_event,final_event


def group(index, success=False):
    return [{'instance_id':f'target{index}','prompt_role':'target','rollout_index':i,
        'reward':int(success and i==0),'completion_tokens':2,'reward_group':index} for i in range(8)]


def state():
    return {'train_tokens':100,'generated_tokens':150,'first_target_success':None,'target_successes':0}


def source():
    return {'candidate_index':1,'source_cohort':'synthetic','initial_target_full_successes':0,
            'source_training_target_successes':0}


def profiles():
    return {'rollouts':512,'counts':[0]*11},{'rollouts':64,'counts':[0]*11}


class EventTests(unittest.TestCase):
    def test_first_success_clock_at_full_group_boundary(self):
        items=group(4)+group(5,True)
        event=first_training_event(state(),items,[[1,1]]*16,3)
        self.assertEqual(event['observation_loss_tokens'],132)
        self.assertEqual(event['generated_tokens_through_group'],182)
        self.assertEqual(event['group_index'],5)

    def test_masked_phase_group_keeps_generation_cost_separate(self):
        s=state();s['train_tokens']=BUDGET-3
        masks=[[1,1],[1,0]]+[[0,0]]*6
        event=first_training_event(s,group(0,True),masks,3)
        self.assertEqual(event['observation_loss_tokens'],BUDGET)
        self.assertEqual(event['generated_tokens_through_group'],166)

    def test_existing_event_not_overwritten(self):
        s=state();s['first_target_success']={'observation_loss_tokens':50,'role':'followon_target_training'}
        first=copy.deepcopy(s['first_target_success'])
        self.assertEqual(first_training_event(s,group(0,True),[[1,1]]*8,3),first)
        self.assertEqual(s['first_target_success'],first)

    def test_scope_and_partial_reward_cannot_be_events(self):
        for key,value in (('prompt_role','scope'),('reward',.5)):
            items=group(0);items[0][key]=value
            with self.assertRaises(ValueError):first_training_event(state(),items,[[1,1]]*8,3)

    def test_omitted_or_mixed_group_rejected(self):
        items=group(0);items[3]['instance_id']='different'
        with self.assertRaises(ValueError):first_training_event(state(),items,[[1,1]]*8,3)
        with self.assertRaises(ValueError):first_training_event(state(),group(0)[:7],[[1,1]]*7,3)

    def test_budget_overrun_rejected(self):
        s=state();s['train_tokens']=BUDGET-1
        with self.assertRaises(ValueError):first_training_event(s,group(0),[[1,1]]*8,3)

    def test_no_event_is_censored_not_failed_forever(self):
        s=state();s['train_tokens']=BUDGET
        event=final_event(source(),s,*profiles())
        self.assertFalse(event['event_observed']);self.assertTrue(event['right_censored'])
        self.assertIsNone(event['first_event']);self.assertEqual(event['observation_loss_tokens'],BUDGET)

    def test_selection_and_completion_count_as_new_events_at_budget(self):
        for role in ('selection','completion'):
            s=state();s['train_tokens']=BUDGET;target,completion=profiles()
            (target if role=='selection' else completion)['counts']=[1]*11
            event=final_event(source(),s,target,completion)
            self.assertTrue(event['event_observed']);self.assertEqual(event['observation_loss_tokens'],BUDGET)
            self.assertEqual(event['first_event']['role'],'followon_target_'+role if role=='selection' else 'followon_completion')

    def test_source_success_not_relabelled_new_followon_success(self):
        s=state();s['train_tokens']=BUDGET;c=source();c['initial_target_full_successes']=1
        event=final_event(c,s,*profiles())
        self.assertTrue(event['prior_source_event_recorded']);self.assertFalse(event['event_observed'])

    def test_incomplete_runs_and_inconsistent_event_refused(self):
        with self.assertRaises(ValueError):final_event(source(),state(),*profiles())
        s=state();s.update(train_tokens=BUDGET,target_successes=1)
        with self.assertRaises(ValueError):final_event(source(),s,*profiles())


if __name__=='__main__':unittest.main()
