"""Small contract tests; all invented raw evidence lives in fixtures/."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from fixture_round import exchange, request
from round_cli import build_plan, canonical, generation_seed_for_sample, score_exchange, sha


class MiniRoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.request = request()
        self.plan = build_plan(self.request, _fixture=True)

    def test_plan_has_three_courses_direct_four_segments_and_17_eval_requests(self):
        self.assertEqual(set(self.plan["branches"]), {"g1", "g2", "g3", "direct"})
        self.assertEqual(len(self.plan["evaluation_requests"]), 17)
        self.assertEqual(self.plan["planned_evaluation_rollouts"], 2048)
        for branch in self.plan["branches"].values():
            self.assertEqual(len(branch["phases"]), 4)
            self.assertEqual(sum(p["quota"] for p in branch["phases"]), 32)
        self.assertTrue(all(p["kind"] == "target_only" for p in self.plan["branches"]["direct"]["phases"]))
        self.assertTrue(all(e["hint"] is None for e in self.plan["evaluation_requests"]))

    def test_challenger_actual_json_must_match_selected_proposal(self):
        self.request["proposal"]["curricula"]["g1"] = self.request["proposal"]["curricula"]["g2"]
        with self.assertRaisesRegex(ValueError, "actual Challenger"):
            build_plan(self.request, _fixture=True)

    def test_hint_without_actual_probe_fails_and_nonhint_stays_na(self):
        stage = self.request["proposal"]["curricula"]["g1"][0]
        stage["kind"] = "hinted_target"
        self.request["challenger"]["raw_output"] = canonical(self.request["proposal"])
        with self.assertRaisesRegex(ValueError, "hint-regret"):
            build_plan(self.request, _fixture=True)

    def test_complete_raw_exchange_scores_gamma_delta_archive_buffer(self):
        value = score_exchange(self.plan, exchange(self.temp.name, self.plan), _fixture=True)
        self.assertEqual(value["summary"]["status"], "complete")
        self.assertEqual(value["summary"]["total_training_generated_tokens"], 128)
        self.assertEqual(value["summary"]["evaluation_rollouts"], 2048)
        self.assertTrue(value["summary"]["at_least_one_curriculum_mixed_update"])
        self.assertEqual(value["round_score"]["kappa"], 6)
        self.assertEqual(set(value["round_score"]["j_plus"]), {"g1", "g2"})
        self.assertEqual(len(value["lineage"]), 17)
        self.assertEqual(len(value["challenger_buffer"]["examples"]), 2)
        self.assertEqual(value["rft_plan"]["epochs"], 1)
        self.assertEqual(value["challenger_buffer"]["completed_updates"], [])
        for breakdown in value["delta"].values():
            self.assertEqual(len(breakdown["deltas"]), 4)
            self.assertAlmostEqual(breakdown["telescoping_residual"], 0.)
            self.assertEqual(breakdown["segments"][-1], "target_only_tail")

    def test_missing_raw_rollout_is_not_dropped_from_denominator(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["evaluations"]["g1_final"]["records"].pop()
        with self.assertRaisesRegex(ValueError, "every frozen"):
            score_exchange(self.plan, supplied, _fixture=True)

    def test_final_raw_condition_must_match_per_class_test_results(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["evaluations"]["g1_final"]["records"][0]["test_outcomes"]["t1"] = False
        with self.assertRaisesRegex(ValueError, "per-class"):
            score_exchange(self.plan, supplied, _fixture=True)

    def test_hint_contamination_wrong_base_or_optimizer_state_rejected(self):
        supplied = exchange(self.temp.name, self.plan)
        for field, value in (("base_checkpoint", "different"), ("optimizer_initial_state_entries", 2), ("beta", .1)):
            edited = copy.deepcopy(supplied)
            edited["branches"]["g1"][field] = value
            with self.assertRaises(ValueError):
                score_exchange(self.plan, edited, _fixture=True)
        supplied["evaluations"]["g1_final"]["records"][0]["hint"] = "bad"
        with self.assertRaisesRegex(ValueError, "unhinted"):
            score_exchange(self.plan, supplied, _fixture=True)

    def test_raw_token_ledger_and_plan_digest_are_binding(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["branches"]["g1"]["phases"][0]["generated_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "matching committed"):
            score_exchange(self.plan, supplied, _fixture=True)
        altered = copy.deepcopy(self.plan)
        altered["B"] += 1
        with self.assertRaisesRegex(ValueError, "intact frozen plan"):
            score_exchange(altered, supplied, _fixture=True)

    def test_default_api_cannot_promote_unit_fixture_evidence(self):
        with self.assertRaisesRegex(ValueError, "fixture stage outcomes"):
            build_plan(self.request)
        with self.assertRaisesRegex(ValueError, "real model evidence"):
            score_exchange(self.plan, exchange(self.temp.name, self.plan))

    def test_common_sampling_seed_excludes_checkpoint_identity(self):
        value = generation_seed_for_sample(self.plan, "instance-1", "sample-1")
        other = copy.deepcopy(self.plan)
        other["base_checkpoint"] = "other"
        self.assertEqual(value, generation_seed_for_sample(other, "instance-1", "sample-1"))
        self.assertNotEqual(value, generation_seed_for_sample(other, "instance-2", "sample-1"))

    def test_unchanged_direct_reuses_actual_base_raw_without_new_samples(self):
        supplied = exchange(self.temp.name, self.plan)
        for req in self.plan["evaluation_requests"]:
            if req["branch"] == "direct":
                supplied["checkpoints"][req["checkpoint_alias"]] = copy.deepcopy(supplied["checkpoints"]["base"])
                supplied["evaluations"][req["alias"]] = {"checkpoint": self.plan["base_checkpoint"],
                    "evaluation_cache_hit": True, "source_evidence_id": "base"}
        value = score_exchange(self.plan, supplied, _fixture=True)
        self.assertEqual(value["summary"]["logical_evaluation_rollout_slots"], 2048)
        self.assertEqual(value["summary"]["evaluation_rollouts"], 1600)
        self.assertEqual(value["summary"]["cached_evaluation_aliases"], 4)
        self.assertTrue(all(not p["checkpoint_weights_changed"] for p in value["budget_audit"]["direct"]))

    def test_cache_requires_actual_source_path_and_original_records(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["evaluations"]["direct_final"] = {"checkpoint": supplied["checkpoints"]["direct_m4"]["path"],
            "evaluation_cache_hit": True, "source_evidence_id": "base"}
        with self.assertRaisesRegex(ValueError, "actually evaluated checkpoint"):
            score_exchange(self.plan, supplied, _fixture=True)

    def test_partial_cache_reuses_p8_then_generates_only_new_24_slots(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["checkpoints"]["g1_m4"] = copy.deepcopy(supplied["checkpoints"]["g1_m3"])
        final = supplied["evaluations"]["g1_final"]
        final["checkpoint"] = supplied["checkpoints"]["g1_m3"]["path"]
        first8 = {(r["instance_id"], r["sample_id"]): r for r in supplied["evaluations"]["g1_m3"]["records"]}
        for index, record in enumerate(final["records"]):
            record["checkpoint"] = final["checkpoint"]
            key = record["instance_id"], record["sample_id"]
            if key in first8:
                final["records"][index] = copy.deepcopy(first8[key])
        final["source_evidence_id"] = "g1_m3"
        final["evaluation_cache_hit"] = False
        value = score_exchange(self.plan, supplied, _fixture=True)
        self.assertEqual(value["evaluation_provenance"]["g1_final"]["rollouts"], 8 * 24)
        self.assertEqual(value["evaluation_provenance"]["g1_final"]["reused_rollout_slots"], 8 * 8)
        self.assertEqual(value["summary"]["evaluation_rollouts"], 2048 - 64)

    def test_cache_source_can_follow_alias_in_plan_order(self):
        supplied = exchange(self.temp.name, self.plan)
        supplied["checkpoints"]["g1_m1"] = copy.deepcopy(supplied["checkpoints"]["g3_m4"])
        supplied["evaluations"]["g1_m1"] = {"checkpoint": supplied["checkpoints"]["g3_m4"]["path"],
            "evaluation_cache_hit": True, "source_evidence_id": "g3_final"}
        value = score_exchange(self.plan, supplied, _fixture=True)
        self.assertEqual(value["summary"]["evaluation_rollouts"], 2048 - 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
