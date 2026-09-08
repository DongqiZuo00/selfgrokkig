import copy
import json
import unittest
from verge_round_core import config
from verge_repair_protocol import (proposal_schema, parse_typed_proposal, finite_inputs, finite_task,
    update_plan, next_group_request, group_token_allocation, challenger_credit, prompt_role,
    assert_challenger_likelihood_is_matched, full_program)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.families = config()["families"]

    def test_schema_disallows_prior_errors(self):
        for family in ("contains_count", "numerical_comparison"):
            proposal = {"stages": [{"family": family, "colors": 2, "length": 2, "mutation": "simple"}]}
            with self.assertRaises(ValueError):
                parse_typed_proposal(json.dumps(proposal), self.families)
            proposal["stages"][0]["mutation"] = "none"
            self.assertEqual(parse_typed_proposal(json.dumps(proposal), self.families), proposal)

    def test_existing_space_kept(self):
        for family in self.families:
            color = 4 if family == "regex_same_num" else 2
            value = {"stages": [{"family": family, "colors": color, "length": 6, "mutation": "none"}]}
            self.assertEqual(parse_typed_proposal(json.dumps(value), self.families), value)

    def test_finite_space_expresses_previous_success_without_template(self):
        # R -> empty; B -> RB. This is a test fixture, NEVER a generated proposal or policy prompt.
        spec = {"family": "finite_map", "colors": 2, "min_input_length": 1,
                "max_input_length": 1, "outputs": ["", "RB"]}
        self.assertEqual(finite_inputs(2, 1, 1), ["R", "B"])
        value = {"stages": [spec]}
        parse_typed_proposal(json.dumps(value), self.families)
        base = {"messages": [{"role": "user", "content": "OFFICIAL DSL\n# Task\nOriginal hard target"}],
                "ground_truth": [{"input": "old"}]}
        saved = copy.deepcopy(base)
        row = finite_task(spec, base, "test_fixture")
        self.assertEqual(base, saved)
        self.assertEqual([c["expected_output"] for c in row["ground_truth"]], ["", "RB"])
        self.assertIn("Correctness outside this finite set is not required", row["messages"][0]["content"])
        self.assertNotIn("Original hard target", row["messages"][0]["content"])

    def test_finite_bad_outputs_rejected(self):
        value = {"stages": [{"family": "finite_map", "colors": 2,
            "min_input_length": 1, "max_input_length": 1, "outputs": ["Y", "RB"]}]}
        with self.assertRaises(ValueError):
            parse_typed_proposal(json.dumps(value), self.families)

    def test_budget_guarantees_training_opportunities(self):
        for n in range(1, 5):
            p = update_plan(n)
            self.assertEqual(p["token_updates"], 32)
            self.assertEqual(p["direct_token_updates"], 32)
            self.assertEqual(sum(s["tokens"] for s in p["stages"]), 524288)
            self.assertGreaterEqual(min(s["token_updates"] for s in p["stages"]), 6)
        with self.assertRaises(ValueError):
            update_plan(3, 131072)

    def test_complete_groups_no_prefix_loss_of_successes(self):
        for remaining in (16384, 8192, 2000, 8, 7, 1):
            p = next_group_request(remaining)
            lengths = [p["max_tokens"]] * 8
            allocation = group_token_allocation(lengths, remaining)
            self.assertLessEqual(sum(allocation), remaining)
            if remaining >= 8:
                self.assertEqual(allocation, lengths)
        with self.assertRaises(ValueError):
            group_token_allocation([2048] * 64, 16384)

    def test_target_mix_and_validity(self):
        self.assertEqual([prompt_role(i, "curriculum") for i in range(4)],
                         ["curriculum", "curriculum", "curriculum", "target"])
        self.assertFalse(challenger_credit([-2, -.00390625, -2], [False, True, False])["update_allowed"])
        self.assertEqual(challenger_credit([0, 0, 0], [True] * 3)["advantages"], [0, 0, 0])
        self.assertTrue(challenger_credit([.1, 0, -.1], [True] * 3)["update_allowed"])

    def test_no_false_masked_grpo_claim(self):
        with self.assertRaises(ValueError):
            assert_challenger_likelihood_is_matched("schema_constrained", "legacy_unmasked_log_probability")
        assert_challenger_likelihood_is_matched("schema_constrained", "same_schema_masked_log_probability")

    def test_prefix_is_generic_not_a_solution(self):
        self.assertEqual(full_program("end\nEND end\n```"),
                         "```manufactoria\nSTART start:\n    NEXT end\nEND end\n```")


if __name__ == "__main__":
    unittest.main()
