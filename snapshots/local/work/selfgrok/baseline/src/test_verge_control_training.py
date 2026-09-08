"""Tiny synthetic gradient tests, not control-experiment results."""
import copy
from types import SimpleNamespace
import unittest
import torch
from common import group_advantages
from verge_control_training import prepare_group
from verge_repair_training import train_step


class ControlTrainingTests(unittest.TestCase):
    def config(self):
        return {"control_only": True, "control_reward_protocol": "appendix_d_v1",
                "frozen_reward_rung": 0, "shaping_coefficient": 1.}

    def records(self):
        return [{"instance_id": "task", "rollout_index": i, "reward": 0,
                 "passed_cases": i % 4, "total_cases": 4,
                 "prompt_token_ids": [0], "completion_token_ids": [1 + i % 3],
                 "completion": str(i)} for i in range(8)]

    def test_primary_configuration_cannot_use_control_rewards(self):
        for cfg in ({}, dict(self.config(), book_suite="verge_book_v2"),
                    dict(self.config(), book_atomic_round=True)):
            with self.assertRaises(ValueError):
                prepare_group(self.records(), {"id":"task"}, {"reward_mode":"dense"}, cfg, role="target")

    def test_optimizer_reward_does_not_overwrite_binary_events_or_actions(self):
        rows = self.records()
        old = copy.deepcopy(rows)
        prepared, adv, result = prepare_group(rows, {"id":"task"}, {"reward_mode":"dense"}, self.config(), role="target")
        self.assertEqual(rows, old)
        self.assertEqual([r["reward"] for r in prepared], [0]*8)
        self.assertEqual([r["completion_token_ids"] for r in prepared], [r["completion_token_ids"] for r in rows])
        self.assertEqual(adv, group_advantages([0,.25,.5,.75]*2, 8))
        self.assertEqual(result["binary_full_successes"], 0)

    def test_binary_control_is_exactly_the_existing_group_convention(self):
        rows = self.records()
        rows[-1].update(reward=1, passed_cases=4)
        _, adv, _ = prepare_group(rows, {"id":"task"}, {"reward_mode":"binary"}, self.config(), role="target")
        self.assertEqual(adv, group_advantages([r["reward"] for r in rows], 8))

    def test_shaping_does_not_evaluate_target_conditions_on_curriculum(self):
        def forbidden(*args):
            raise AssertionError("Target check leaked into an intermediate reward")
        _, adv, result = prepare_group(self.records(), {"id":"task"}, {"reward_mode":"target_condition_shaped"},
                                      self.config(), role="curriculum", condition_evaluator=forbidden)
        self.assertEqual(adv, [0.]*8)
        self.assertEqual(result["mean_optimization_reward"], 0.)

    def test_shaping_target_keeps_zero_complete_success_count(self):
        _, adv, result = prepare_group(self.records(), {"id":"task"}, {"reward_mode":"target_condition_shaped"},
            self.config(), role="target", condition_evaluator=lambda text,row: [int(text)%2,0])
        self.assertTrue(any(adv))
        self.assertEqual(result["binary_full_successes"], 0)

    def test_malformed_group_rejected(self):
        for rows in (self.records()[:7], list(reversed(self.records()))):
            with self.assertRaises(ValueError):
                prepare_group(rows, {"id":"task"}, {"reward_mode":"dense"}, self.config(), role="target")

    def test_dense_signal_reaches_optimizer_without_fabricating_full_success(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4,4))
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weight[input_ids])
        for mode, expect_step in (("binary", False), ("dense", True)):
            model = Tiny()
            opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
            rows, adv, event = prepare_group(self.records(), {"id":"task"}, {"reward_mode": mode}, self.config(), role="target")
            before = model.weight.detach().clone()
            metric = train_step(model, None, opt, scheduler, rows, adv, [[1]]*8, {"kl_beta":0.}, 0)
            self.assertEqual(metric["optimizer_step"], expect_step)
            self.assertEqual(not torch.equal(before, model.weight), expect_step)
            self.assertEqual(event["binary_full_successes"], 0)


if __name__ == "__main__":
    unittest.main()
