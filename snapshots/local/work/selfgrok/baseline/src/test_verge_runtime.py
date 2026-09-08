"""CPU-only implementation tests, no model sampling or target auditing."""
import copy
import unittest
from verge_round_core import condition_vector, condition_names
from verge_round_decision import decide, paired_interval
from test_verge_round import ProtocolTest


def endpoint(value, scope=1.0, changed=True):
    vector = [value] * 11
    return {"checkpoint": "mock", "training": {"train_tokens": 10, "optimizer_steps": int(changed)},
        "scope": {"full_pass_rate": scope, "mean_case_fraction": scope},
        "target": {"counts": [value * 8] * 11, "rates": vector,
                   "keyed_vectors": {f"i{i}::{j}": vector for i in range(2) for j in range(4)}}}


class RuntimeTest(unittest.TestCase):
    def test_verifier_ladder(self):
        from validate_v5_3_conditional_route import PROGRAMS
        cases = [{"input": inp, "expected_output": out, "expected_accepted": True, "check_output": True}
                 for inp, out in [("B", "RB"), ("R", "RR")]]
        row = {"ground_truth": cases}
        self.assertEqual(condition_vector(PROGRAMS[("R", "prepend")], row), [1] * 11)
        v = condition_vector(PROGRAMS[("R", "branch_consume_same")], row)
        self.assertEqual(v[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(v, v[1:])))
        self.assertEqual(condition_vector("not a program", row)[-1], 0)

    def frozen(self):
        return {"config": {"train_tokens_per_branch": 10, "reward_switch_q": 4,
                "bootstrap_replicates": 100, "scope_max_drop": .05, "selector_beta": 2},
                "initial": endpoint(0), "reward_condition": 1, "selection_draw": .1,
                "generation": {"candidates": [{"index": i, "valid": True} for i in (1, 2, 3)]}}

    def test_all_zero_keeps_start(self):
        b = {i: endpoint(0, changed=False) for i in range(4)}
        d = decide(self.frozen(), b)
        self.assertEqual(d["selected"]["name"], "start")
        self.assertEqual(d["challenger_advantages"], [0, 0, 0])
        self.assertTrue(d["direct_start_deduplicated"])

    def test_reward_switch_and_scope_filter(self):
        b = {0: endpoint(0), 1: endpoint(1), 2: endpoint(1, scope=.1), 3: endpoint(0)}
        d = decide(self.frozen(), b)
        self.assertEqual(d["reward_rung_zero_based"], 10)
        self.assertEqual(d["selected"]["name"], "candidate_1")
        self.assertFalse(d["evaluated_candidates"][1]["eligible"])
        b[2] = endpoint(0)
        d = decide(self.frozen(), b)
        self.assertEqual(d["reward_rung_zero_based"], 1)
        self.assertEqual(d["selection_rung_zero_based"], 10)

    def test_paired_interval(self):
        self.assertEqual(paired_interval(endpoint(1)["target"], endpoint(0)["target"], 0, 100), [1, 1])


if __name__ == "__main__":
    unittest.main()
