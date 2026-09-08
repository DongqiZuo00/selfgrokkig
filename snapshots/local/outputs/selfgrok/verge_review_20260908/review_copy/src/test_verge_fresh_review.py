"""Independent CPU regressions; no model checkpoint loading or sampling."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from verge_repair_training import train_step
from verge_round_core import profile, centered_advantages

def equal_tree(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.zeros(4,4))
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.weight[input_ids])

class FreshReview(unittest.TestCase):
    def test_empty_test_sets_rejected_before_program_parsing(self):
        from common import verify_full, verify_all_tests
        from verge_round_core import condition_vector
        for verify in (verify_full, verify_all_tests):
            with self.assertRaises(ValueError): verify("not a program", [])
        with self.assertRaises(ValueError): condition_vector("not a program", {"ground_truth":[]})
    def test_missing_expected_output_is_not_an_empty_output(self):
        from common import verify_full, verify_all_tests
        end="START start:\n    NEXT end\nEND end"
        for verify in (verify_full, verify_all_tests):
            with self.assertRaises(ValueError):verify(end,[{"input":""}])
            self.assertEqual(verify(end,[{"input":"","expected_output":""}]).reward,1)
            self.assertEqual(verify(end,[{"input":"","check_output":False,"expected_accepted":True}]).reward,1)
    def test_invalid_threshold_denominators_rejected(self):
        from verge_round_core import thresholds
        for n in (0,-1,True,1.5,None):
            with self.assertRaises(ValueError):thresholds(n)
        self.assertEqual(thresholds(20),[.05,.25,.5,.75])
    def test_constant_reward_preserves_warm_optimizer_and_scheduler(self):
        for reward in (0,1):
            model=Tiny()
            opt=torch.optim.AdamW(model.parameters(),lr=.01,betas=(.9,.95),weight_decay=0.)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:1/(i+1))
            items=[{"prompt_token_ids":[0],"completion_token_ids":[1,2]} for _ in range(8)]
            items[0]["completion_token_ids"]=[3,3]
            masks=[[1,1] for _ in items]
            real=train_step(model,None,opt,sched,items,centered_advantages([1]+[0]*7),masks,{"kl_beta":0.},0)
            self.assertTrue(real["optimizer_step"])
            weights=copy.deepcopy(model.state_dict())
            opt_state=copy.deepcopy(opt.state_dict())
            sched_state=copy.deepcopy(sched.state_dict())
            metric=train_step(model,None,opt,sched,items,centered_advantages([reward]*8),masks,{"kl_beta":0.},1)
            with self.subTest(reward=reward):
                self.assertFalse(metric["optimizer_step"])
                self.assertTrue(equal_tree(weights,model.state_dict()))
                self.assertTrue(equal_tree(opt_state,opt.state_dict()))
                self.assertTrue(equal_tree(sched_state,sched.state_dict()))
    def test_fresh_constant_reward_preserves_empty_optimizer(self):
        for reward in (0,1):
            model=Tiny()
            opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:1.)
            items=[{"prompt_token_ids":[0],"completion_token_ids":[1]}]*8
            before=copy.deepcopy(model.state_dict())
            metric=train_step(model,None,opt,sched,items,centered_advantages([reward]*8),[[1]]*8,{"kl_beta":0.},0)
            self.assertFalse(metric["optimizer_step"])
            self.assertTrue(equal_tree(before,model.state_dict()))
            self.assertEqual(opt.state_dict()["state"],{})
    def test_legacy_nonzero_kl_still_explicitly_steps(self):
        model=Tiny(); ref=Tiny()
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        items=[{"prompt_token_ids":[0],"completion_token_ids":[1]}]*8
        result=train_step(model,ref,opt,sched,items,[0.]*8,[[1]]*8,{"kl_beta":.02},1)
        self.assertTrue(result["optimizer_step"]) # Legacy protocol is preserved, not approved as latest design.
    def test_profile_rejects_duplicate_rollout_identity(self):
        rows=[{"id":"a"}]
        records=[{"instance_id":"a","rollout_index":0,"reward":0,"completion":"x"}]*2
        with patch("verge_round_core.condition_vector",return_value=[1]+[0]*10):
            with self.assertRaises(ValueError):profile(records,rows)
    def test_profile_rejects_duplicate_instance_identity(self):
        rows=[{"id":"a"},{"id":"a"}]
        records=[{"instance_id":"a","rollout_index":0,"reward":0,"completion":"x"}]
        with patch("verge_round_core.condition_vector",return_value=[1]+[0]*10):
            with self.assertRaises(ValueError):profile(records,rows)
    def test_profile_rejects_empty_samples(self):
        with self.assertRaises(ValueError):profile([],[])
    def test_profile_rejects_missing_instances_and_unbalanced_draws(self):
        for records in [
            [{"instance_id":"a","rollout_index":0,"reward":0,"completion":"x"}],
            [{"instance_id":"a","rollout_index":0,"reward":0,"completion":"x"},
             {"instance_id":"b","rollout_index":1,"reward":0,"completion":"x"}]]:
            with patch("verge_round_core.condition_vector",return_value=[1]+[0]*10):
                with self.assertRaises(ValueError):profile(records,[{"id":"a"},{"id":"b"}])
    def test_profile_preserves_valid_counts(self):
        records=[{"instance_id":x,"rollout_index":i,"reward":0,"completion":"x"}
                 for x in ("a","b") for i in (0,1)]
        with patch("verge_round_core.condition_vector",return_value=[1]+[0]*10):
            p=profile(records,[{"id":"a"},{"id":"b"}])
        self.assertEqual(p["counts"],[4]+[0]*10)
        self.assertEqual(p["rollouts"],4)
if __name__=="__main__":unittest.main()
