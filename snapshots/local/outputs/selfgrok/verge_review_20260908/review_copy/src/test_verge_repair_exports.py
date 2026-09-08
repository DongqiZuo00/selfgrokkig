"""Synthetic formatting tests, not model observations or scientific rewards."""
import copy
import unittest
from verge_repair_exports import result_rows, condition_rows, latex_table


class ResultExportTest(unittest.TestCase):
    def setUp(self):
        profile = {"rollouts": 16, "counts": [8] * 10 + [1],
                   "rates": [.5] * 10 + [1 / 16], "condition_names": [f"c_{i}" for i in range(11)]}
        endpoint = {"target": profile, "checkpoint": "example", "scope": {
            "rollouts": 8, "full_pass_count": 2, "full_pass_rate": .25, "mean_case_fraction": .5}}
        self.frozen = {"initial": endpoint}
        self.branches = {i: copy.deepcopy(endpoint) for i in range(4)}
        for branch in self.branches.values():
            branch["training"] = {"train_tokens": 1000, "optimizer_steps": 2,
                                  "update": 3, "nonzero_advantage_tokens": 400}
        self.decision = {"reward_rung_zero_based": 0, "selection_rung_zero_based": 10,
                         "selected": {"name": "start"}, "evaluated_candidates": []}
        for i in range(1, 4):
            self.decision["evaluated_candidates"].append({"index": i, "reward": 0.,
                "observed_endpoint_reward": .125, "selection_gain": 0.,
                "observed_endpoint_gain": -.125, "paired_95_interval": [0., 0.],
                "identical_policy_by_zero_update_provenance": True, "eligible": False, "scope_safe": True})

    def test_missing_is_not_zero(self):
        rows = result_rows(self.frozen, self.branches, self.decision)
        self.assertIsNone(rows[0]["credit_gain"])
        self.assertIsNone(rows[1]["selection_gain"])
        self.assertEqual(rows[2]["credit_gain"], 0.)

    def test_identity_keeps_observed_noise(self):
        row = result_rows(self.frozen, self.branches, self.decision)[2]
        self.assertEqual(row["raw_observed_credit_gain"], .125)
        self.assertEqual(row["raw_observed_selection_gain"], -.125)
        self.assertEqual(row["selection_ci95_low"], 0.)

    def test_all_conditions_and_distinct_rungs(self):
        flat = condition_rows(self.frozen, self.branches)
        self.assertEqual(len(flat), 55)
        row = result_rows(self.frozen, self.branches, self.decision)[0]
        self.assertEqual(row["credit_condition_count"], 8)
        self.assertEqual(row["selection_condition_count"], 1)

    def test_table_escapes_and_denominators(self):
        table = latex_table(result_rows(self.frozen, self.branches, self.decision),
                            self.decision, {"counts": [0], "rollouts": 4})
        self.assertIn(r"candidate\_1", table)
        self.assertIn("1/16", table)
        self.assertIn("0/4", table)
        self.assertIn("1000 loss tokens", table)


if __name__ == "__main__":
    unittest.main()
