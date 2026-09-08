"""Synthetic fixtures only; no diagnostic model samples or new training seeds."""
import copy
import unittest
from verge_control_rewards import checked_verdict, solver_rewards, direct_reward_phases, scalar_partial_credit
from verge_independent_sampling import PROTOCOL


def record(passed, total=4):
    return {"reward": int(passed == total), "passed_cases": passed, "total_cases": total}


class ControlRewardTests(unittest.TestCase):
    def test_dense_is_fraction_not_partial_count(self):
        rows = [record(0), record(1), record(3), record(4)]
        self.assertEqual(solver_rewards("dense", rows, role="target"), [0., .25, .75, 1.])
        self.assertEqual(solver_rewards("binary", rows, role="target"), [0., 0., 0., 1.])

    def test_all_intermediate_rewards_remain_binary(self):
        for mode in ("binary", "dense", "target_condition_shaped"):
            self.assertEqual(solver_rewards(mode, [record(2), record(4)], role="curriculum"), [0., 1.])
        with self.assertRaises(ValueError):
            solver_rewards("target_condition_shaped", [record(2)], role="curriculum", condition_vectors=[[1,0]])

    def test_shaping_uses_same_explicit_rung_and_keeps_full_pass(self):
        rows = [record(0), record(2), record(4)]
        vectors = [[1,0,0], [1,1,0], [1,1,1]]
        self.assertEqual(solver_rewards("target_condition_shaped", rows, role="target",
            condition_vectors=vectors, frozen_rung=1, coefficient=1.), [0., 1., 2.])

    def test_shaping_configuration_must_be_predefined(self):
        for kwargs in ({}, {"frozen_rung": 0}, {"frozen_rung": 0, "coefficient": float("nan")},
                       {"frozen_rung": 0, "coefficient": -1.}):
            with self.assertRaises(ValueError):
                solver_rewards("target_condition_shaped", [record(0)], role="target",
                               condition_vectors=[[1,0]], **kwargs)

    def test_non_nested_or_false_full_vectors_are_rejected(self):
        for vector in ([0,1,0], [1,1], [1,.5,0], []):
            with self.assertRaises(ValueError):
                solver_rewards("target_condition_shaped", [record(0)], role="target",
                               condition_vectors=[vector], frozen_rung=0, coefficient=1.)

    def test_inconsistent_verifier_counts_are_rejected(self):
        for row in (record(0,0), {"reward": 0, "passed_cases": 4, "total_cases": 4},
                    {"reward": 1, "passed_cases": 0, "total_cases": 4}, record(5), record(-1)):
            with self.assertRaises(ValueError):
                checked_verdict(row)

    def test_dense_warmup_has_explicit_loss_token_boundary(self):
        phases = direct_reward_phases("dense_then_binary", 524288, warmup_loss_tokens=131072)
        self.assertEqual([p["reward_mode"] for p in phases], ["dense", "binary"])
        self.assertEqual(sum(p["tokens"] for p in phases), 524288)
        for boundary in (None, 0, 524288, 900000):
            with self.assertRaises(ValueError):
                direct_reward_phases("dense_then_binary", 524288, warmup_loss_tokens=boundary)

    def batch(self, passed):
        return {"data_role": "fixed_checkpoint_endpoint", "checkpoint": "test_only",
            "records": [dict(record(n), sampling_protocol=PROTOCOL, instance_id=str(i),
                rollout_index=0, sampling_child_seed=42+i) for i,n in enumerate(passed)]}

    def test_scalar_credit_is_endpoint_fraction_difference(self):
        result = scalar_partial_credit(self.batch([2,4]), self.batch([0,2]))
        self.assertEqual(result["reward"], .5)
        result = scalar_partial_credit(self.batch([2,4]), self.batch([0,2]), unchanged_from_common_start=True)
        self.assertEqual(result["reward"], 0.)
        self.assertEqual(result["observed_gain"], .5)

    def test_training_pool_and_misaligned_endpoint_are_rejected(self):
        a, b = self.batch([2,4]), self.batch([0,2])
        for bad in (dict(a, data_role="training_trajectory"), dict(a, checkpoint=None)):
            with self.assertRaises(ValueError):
                scalar_partial_credit(bad, b)
        bad = copy.deepcopy(a)
        bad["records"][0]["sampling_child_seed"] = 999
        with self.assertRaises(ValueError):
            scalar_partial_credit(bad, b)
        bad = copy.deepcopy(a)
        bad["records"][1]["sampling_child_seed"] = bad["records"][0]["sampling_child_seed"]
        with self.assertRaises(ValueError):
            scalar_partial_credit(bad, b)
        bad = copy.deepcopy(a)
        bad["records"].append(bad["records"][0])
        with self.assertRaises(ValueError):
            scalar_partial_credit(bad, b)


if __name__ == "__main__":
    unittest.main()
