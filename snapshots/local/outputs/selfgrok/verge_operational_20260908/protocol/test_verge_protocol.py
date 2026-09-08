"""Synthetic contract tests only. Fixtures are not benchmark classes or results."""

import unittest
from dataclasses import replace

from verge_protocol import (
    Candidate, CheckpointArchive, Evaluation, StageSpec, TestClasses,
    decompose_stages, frontier, paired_gain, plan_round, score_round,
)


def ev(checkpoint, row=(32, 0, 0, 0, 0, 0), *, rows=None, n=32):
    return Evaluation(checkpoint, 7, "SYNTHETIC-selection-v1", ("i0", "i1"),
                      tuple(f"r{i}" for i in range(n)), tuple(rows or (row, row)))


def stage(kind="registered"):
    return StageSpec(kind, {"fixture": True})


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.classes = TestClasses(tuple("abcdef"), {t: "fixture_A" if t in "ab" else "fixture_B" for t in "abcdef"})

    def test_min_class_not_pooled_rate_and_six_nested_conditions(self):
        result = self.classes.evaluate(True, dict(zip("abcdef", (True, False, True, True, True, False))))
        self.assertEqual(result.class_rates, {"fixture_A": .5, "fixture_B": .75})
        self.assertEqual(result.weakest_class_rate, .5)
        self.assertEqual(result.conditions, (True, True, True, True, False, False))
        self.assertEqual(result.binary_reward, 0)

    def test_timeout_and_missing_output_retain_denominators(self):
        result = self.classes.evaluate(True, dict(zip("abcdef", (True, None, True, True, True, True))))
        self.assertEqual(result.class_rates["fixture_A"], .5)
        self.assertEqual(result.class_rates["fixture_B"], 1)
        with self.assertRaises(ValueError):
            self.classes.evaluate(True, {"a": True})

    def test_full_pass_binary_reward_and_parse_fail(self):
        passed = self.classes.evaluate(True, {t: True for t in "abcdef"})
        self.assertEqual(passed.binary_reward, 1)
        self.assertTrue(all(passed.conditions))
        failed = self.classes.evaluate(False, {t: None for t in "abcdef"})
        self.assertFalse(any(failed.conditions))
        with self.assertRaises(ValueError):
            self.classes.evaluate(False, {t: True for t in "abcdef"})

    def test_incomplete_blank_duplicate_or_too_small_manifest_rejected(self):
        for ids, mapping in [(("a", "b", "c", "d"), {"a": "x"}),
                             (("a", "b", "c", "d"), dict.fromkeys("abcd", "")),
                             (("a", "a", "c", "d"), dict.fromkeys("acd", "x")),
                             (("a", "b", "c"), dict.fromkeys("abc", "x"))]:
            with self.subTest(ids=ids, mapping=mapping), self.assertRaises(ValueError):
                TestClasses(ids, mapping)

    def test_declared_class_universe_cannot_lose_an_entire_class(self):
        ids = tuple("abcd")
        with self.assertRaises(ValueError):
            TestClasses(ids, dict.fromkeys(ids, "A"), expected_classes=("A", "B"))
        complete = TestClasses(ids, dict(zip(ids, ("A", "A", "B", "B"))), expected_classes=("A", "B"))
        self.assertEqual(set(complete.class_by_test.values()), {"A", "B"})


class BranchPlanningTests(unittest.TestCase):
    def test_shared_base_fresh_optimizer_binary_loss_and_generated_token_budget(self):
        plan = plan_round("base-ckpt", {"g1": [stage()], "g2": [stage("hinted_target")]}, 100, .2)
        for branch in plan.curricula + (plan.direct,):
            self.assertEqual(branch.base_checkpoint, "base-ckpt")
            self.assertTrue(branch.fresh_optimizer)
            self.assertEqual((branch.beta, branch.weight_decay), (0, 0))
            self.assertEqual(branch.solver_reward, "binary_full_pass")
            self.assertEqual(sum(branch.stage_tokens) + branch.tail_tokens, 100)
        self.assertEqual(plan.budget_unit, "generated_tokens")

    def test_direct_saves_union_of_course_boundaries_including_tail(self):
        plan = plan_round("base", {"a": [stage()] * 2, "b": [stage()] * 3}, 101, .2)
        self.assertEqual(plan.curricula[0].stage_tokens, (41, 40))
        self.assertEqual(plan.curricula[0].tail_tokens, 20)
        self.assertEqual(plan.direct.milestones, (0, 27, 41, 54, 81, 101))
        self.assertEqual(plan.curricula[0].milestones[-2:], (81, 101))

    def test_three_stage_interfaces_are_preserved(self):
        plan = plan_round("base", {"a": [stage(k) for k in ("registered", "tape_transform", "hinted_target")]}, 100, .25)
        self.assertEqual([s.kind for s in plan.curricula[0].stages], ["registered", "tape_transform", "hinted_target"])
        with self.assertRaises(ValueError):
            StageSpec("teacher_distillation", {"fixture": True})

    def test_invalid_budget_and_reserved_direct_name_rejected(self):
        for B, eta in ((0, .2), (2, .2), (100, 0), (100, 1)):
            with self.subTest(B=B, eta=eta), self.assertRaises(ValueError):
                plan_round("base", {"a": [stage()]}, B, eta)
        with self.assertRaises(ValueError):
            plan_round("base", {"direct": [stage()]}, 100, .2)


class EvaluationAndBootstrapTests(unittest.TestCase):
    def test_counts_are_empirical_nested_and_complete(self):
        for row in ((32, 33, 0, 0, 0, 0), (32, 1, 2, 0, 0, 0), (32, 0, 0, 0, 0), (32, .5, 0, 0, 0, 0)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                ev("bad", row)
        with self.assertRaises(ValueError):
            replace(ev("bad"), counts=((32, 0, 0, 0, 0, 0),))

    def test_frontier_saturation_and_success_count(self):
        self.assertEqual(frontier(ev("base")), 2)
        self.assertEqual(frontier(ev("bad", (0, 0, 0, 0, 0, 0))), 1)
        self.assertEqual(frontier(ev("saturated", (32, 32, 32, 32, 32, 29))), 7)
        self.assertEqual(ev("successful", (32, 32, 32, 32, 32, 4)).success_count, 8)

    def test_paired_bootstrap_resamples_instances_not_rollouts(self):
        candidate = ev("g", rows=((32, 32, 0, 0, 0, 0), (32, 0, 0, 0, 0, 0)))
        gain = paired_gain(candidate, ev("direct"), 2, resamples=1000)
        self.assertEqual((gain.estimate, gain.ci_low, gain.ci_high, gain.n_instances), (.5, 0, 1, 2))
        self.assertFalse(gain.positive)  # positive mean alone is insufficient
        self.assertEqual(gain, paired_gain(candidate, ev("direct"), 2, resamples=1000))

    def test_cross_seed_manifest_instance_or_sample_mismatch_rejected(self):
        base = ev("base")
        altered = [replace(base, training_seed=8), replace(base, manifest_id="heldout"),
                   replace(base, instance_ids=("i1", "i0")), replace(base, sample_ids=tuple(f"z{i}" for i in range(32)))]
        for other in altered:
            with self.subTest(other=other), self.assertRaises(ValueError):
                paired_gain(other, base, 2)


class RoundScoringTests(unittest.TestCase):
    def test_full_curriculum_gain_and_positive_ci_enter_j_plus(self):
        score = score_round(ev("base"), ev("direct"), {"good": ev("good", (32, 16, 0, 0, 0, 0)), "zero": ev("zero")}, q=4)
        self.assertEqual(score.kappa, 2)
        self.assertFalse(score.reward_switched)
        self.assertEqual(score.gains["good"].estimate, .5)
        self.assertEqual(score.j_plus, ("good",))

    def test_reward_switch_requires_two_branches_and_includes_direct(self):
        direct = ev("direct", (32, 32, 32, 32, 32, 4))
        good = ev("good", (32, 32, 32, 32, 32, 8))
        no_switch = score_round(ev("base"), direct, {"good": good}, q=12)
        self.assertEqual((no_switch.kappa, no_switch.reward_switched), (2, False))
        switched = score_round(ev("base"), direct, {"good": good}, q=8)
        self.assertEqual((switched.kappa, switched.reward_switched), (6, True))
        self.assertEqual(switched.gains["good"].estimate, .125)

    def test_one_round_kappa_is_frozen_for_all_curricula(self):
        direct = ev("direct", (32, 32, 32, 32, 32, 4))
        score = score_round(ev("base"), direct, {
            "good": ev("good", (32, 32, 32, 32, 32, 8)),
            "no_success": ev("other", (32, 32, 16, 0, 0, 0)),
        }, q=8)
        self.assertEqual(score.kappa, 6)
        self.assertEqual(score.gains["no_success"].estimate, -.125)
        self.assertNotIn("no_success", score.j_plus)

    def test_final_batch_size_and_q_are_enforced(self):
        with self.assertRaises(ValueError):
            score_round(ev("base"), ev("direct"), {"g": ev("g")}, q=0)
        probe = ev("probe", (8, 0, 0, 0, 0, 0), n=8)
        with self.assertRaises(ValueError):
            score_round(probe, probe, {"g": probe}, q=1)


class ArchiveTests(unittest.TestCase):
    def test_zero_parse_valid_has_explicit_zero_count_and_structure_zero(self):
        candidate = Candidate(ev("unparsed", (0, 0, 0, 0, 0, 0)), 0)
        self.assertEqual((candidate.key, candidate.evaluation.parse_valid_count), ((1, 0), 0))
        with self.assertRaises(ValueError):
            Candidate(candidate.evaluation, 1)

    def test_cycle_fraction_half_is_structure_one(self):
        self.assertEqual(Candidate(ev("loop"), 32).structure, 1)
        self.assertEqual(Candidate(ev("less_loop"), 31).structure, 0)

    def test_empty_cells_preserve_detours_and_lead_uses_highest_rung(self):
        archive = CheckpointArchive()
        lead = Candidate(ev("lead", (32, 32, 8, 0, 0, 0)), 0)
        detour = Candidate(ev("detour", (32, 16, 0, 0, 0, 0)), 40)
        self.assertEqual(archive.consider(lead), "inserted")
        self.assertEqual(archive.consider(detour), "inserted")
        self.assertEqual(len(archive.cells), 2)
        self.assertEqual(archive.lead, lead)
        self.assertIn(detour.key, archive.cells)

    def test_same_cell_requires_significant_gain_and_exact_tie_keeps_incumbent(self):
        archive = CheckpointArchive()
        initial = Candidate(ev("initial", (32, 8, 0, 0, 0, 0)), 0)
        improved = Candidate(ev("improved", (32, 16, 0, 0, 0, 0)), 0)
        archive.consider(initial)
        self.assertEqual(archive.consider(improved), "replaced")
        self.assertEqual(archive.consider(Candidate(ev("tie", (32, 16, 0, 0, 0, 0)), 0)), "retained")
        self.assertEqual(archive.lead, improved)

    def test_saturated_terminal_cell_compares_full_pass_rate(self):
        archive = CheckpointArchive()
        archive.consider(Candidate(ev("sat", (32, 32, 32, 32, 32, 29)), 0))
        best = Candidate(ev("better", (32, 32, 32, 32, 32, 31)), 0)
        self.assertEqual(archive.consider(best), "replaced")
        self.assertEqual(archive.lead.key, (7, 0))
        self.assertTrue(archive.saturated)


class StageDecompositionTests(unittest.TestCase):
    def fixture(self):
        base = ev("base", (8, 0, 0, 0, 0, 0), n=8)
        course = [base, ev("g_stage", (8, 6, 0, 0, 0, 0), n=8), ev("g_final", (8, 4, 0, 0, 0, 0), n=8)]
        direct = [base, ev("d_stage", (8, 0, 0, 0, 0, 0), n=8), ev("d_final", (8, 1, 0, 0, 0, 0), n=8)]
        final_g = ev("g_final", (32, 20, 0, 0, 0, 0))
        final_d = ev("d_final", (32, 4, 0, 0, 0, 0))
        return course, direct, final_g, final_d

    def test_tail_can_have_negative_delta_and_correction_is_explicit(self):
        result = decompose_stages(*self.fixture(), (0, 80, 100), 2)
        self.assertEqual(result.deltas, (.75, -.375))
        self.assertEqual(result.sampling_correction, .125)
        self.assertEqual(result.gamma, .5)
        self.assertEqual(sum(result.deltas) + result.sampling_correction, result.gamma)
        self.assertEqual(result.telescoping_residual, 0)

    def test_stage_pairing_sample_prefix_checkpoint_and_base_are_checked(self):
        course, direct, fg, fd = self.fixture()
        variants = [([replace(course[0], checkpoint="other-base")] + course[1:], direct, fg, fd),
                    (course, direct, replace(fg, checkpoint="different-final"), fd),
                    (course, [direct[0], replace(direct[1], training_seed=9), direct[2]], fg, fd),
                    ([course[0], replace(course[1], sample_ids=tuple(f"z{i}" for i in range(8))), course[2]], direct, fg, fd)]
        for values in variants:
            with self.subTest(values=values), self.assertRaises(ValueError):
                decompose_stages(*values, (0, 80, 100), 2)

    def test_missing_tail_or_impossible_probe_subset_rejected(self):
        course, direct, fg, fd = self.fixture()
        with self.assertRaises(ValueError):
            decompose_stages(course[:2], direct[:2], fg, fd, (0, 80), 2)
        with self.assertRaises(ValueError):
            decompose_stages(course, direct, replace(fg, counts=((32, 1, 0, 0, 0, 0),) * 2), fd, (0, 80, 100), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
