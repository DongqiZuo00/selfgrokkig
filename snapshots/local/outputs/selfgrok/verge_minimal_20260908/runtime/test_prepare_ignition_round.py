"""CPU-only fixtures and read-only real evidence checks; no model experiment."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_ignition_round as p
import prepare_named_round as named
import test_execute_round as executor_fixture
from execute_round import execute_base, build_baseline_plan


class IgnitionPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = named.load_view(named.VIEW)
        cls.real_screen = named.read(p.ROOT / "evidence/q_ignition_review_20260908/verified_screen.json")
        cls.real_q = __import__("json").loads(named.read(p.ROOT / "evidence/q_ignition_review_20260908/challenger_sample_0.json")["raw_output"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ignition_unit_fixture_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def baseline_fixture(self):
        helper = executor_fixture.ExecutorTests(methodName="test_base_only_batches_are_8_and_seeds_are_chunk_not_per_sample")
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        prep = self.root / "SYNTHETIC_UNIT_PREPARATION"
        prep.mkdir()
        baseline = build_baseline_plan(base_checkpoint=helper.plan["base_checkpoint"],
            training_seed=helper.plan["training_seed"], selection=helper.plan["selection"])
        fragment = execute_base(baseline, helper.rows, prep / "base", solver_factory=helper.factory,
                                fingerprint=helper.fingerprint, _fixture=True)
        named.write_json(prep / "plan.json", helper.plan)
        named.write_json(prep / "READY.json", {"status": "real_base_and_challenger_complete",
            "plan_sha256": helper.plan["plan_sha256"], "baseline_contract_sha256": baseline["baseline_contract_sha256"]})
        return prep, baseline, fragment

    def test_observed_prior_uses_actual_concise_mixed_stages(self):
        eligible, observations = p.observed_starts(self.loaded["stages"], self.real_screen)
        self.assertEqual(eligible, ["identity", "append_R"])
        self.assertEqual([(o["successes"], o["n"]) for o in observations], [(1, 8), (1, 8)])

    def test_schema_changes_only_g1_first_position(self):
        stages = self.loaded["stages"]
        original = named.proposal_schema(stages)
        modified = p.ignition_schema(stages, ["identity", "append_R"])
        expected = copy.deepcopy(original)
        expected["properties"]["curricula"]["properties"]["g1"] = modified["properties"]["curricula"]["properties"]["g1"]
        self.assertEqual(expected, modified)
        first, second, third = modified["properties"]["curricula"]["properties"]["g1"]["prefixItems"]
        self.assertEqual(len(first["oneOf"]), 2)
        self.assertEqual(second, original["properties"]["curricula"]["properties"]["g1"]["items"])
        self.assertEqual(second, third)
        self.assertEqual(len(second["oneOf"]), 9)

    def test_training_variant_must_match_actual_screen(self):
        stages = copy.deepcopy(self.loaded["stages"])
        stages[0]["rows"] = stages[0]["rows_by_variant"]["official"]
        with self.assertRaisesRegex(ValueError, "prompts differ"):
            p.observed_starts(stages, self.real_screen)
        screen = copy.deepcopy(self.real_screen)
        screen["summaries"][0]["mixed_group"] = False
        with self.assertRaisesRegex(ValueError, "two actual"):
            p.observed_starts(self.loaded["stages"], screen)

    def test_actual_old_q_is_preserved_and_would_not_satisfy_new_prior(self):
        before = copy.deepcopy(self.real_q)
        with self.assertRaisesRegex(ValueError, "first-stage prior"):
            p.assert_proposal(self.real_q, self.loaded["stages"], ["identity", "append_R"], self.real_q["base_checkpoint"])
        self.assertEqual(self.real_q, before)

    def test_schema_acceptance_does_not_fix_other_eight_slots(self):
        by_id = {s["stage_id"]: {k: s[k] for k in ("stage_id", "kind", "spec")} for s in self.loaded["stages"]}
        # These JSON objects are schema fixtures, never generated proposals.
        for first in ("identity", "append_R"):
            for other in by_id:
                fixture = {"base_checkpoint": "SYNTHETIC_UNIT_BASE", "curricula": {
                    "g1": [by_id[first], by_id[other], by_id[other]],
                    "g2": [by_id[other]] * 3, "g3": [by_id[other]] * 3}}
                p.assert_proposal(fixture, self.loaded["stages"], ["identity", "append_R"], "SYNTHETIC_UNIT_BASE")

    def test_cross_preparation_cache_preserves_all_raw_paths_hashes_records(self):
        prep, baseline, original = self.baseline_fixture()
        _, fragment, report = p.reuse_base(prep, baseline, _fixture=True)
        self.assertEqual(fragment["evaluations"], original["evaluations"])
        self.assertEqual(fragment["checkpoints"], original["checkpoints"])
        self.assertEqual(report["new_evaluation_rollouts"], 0)
        self.assertEqual(report["new_evaluation_generated_tokens"], 0)
        self.assertEqual(report["reused_rollout_slots"], 256)
        self.assertEqual(len(report["raw_sources"]), 32)
        self.assertEqual(fragment["original_execution"], original["execution"])
        self.assertEqual(fragment["execution"]["actual_generation_tokens"], 0)

    def test_cache_rejects_changed_contract_or_raw_file(self):
        prep, baseline, original = self.baseline_fixture()
        changed = copy.deepcopy(baseline)
        changed["baseline_contract_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "different baseline contract"):
            p.reuse_base(prep, changed, _fixture=True)
        first = original["evaluations"]["base"]["records"][0]
        Path(first["generation_evidence_id"].split("#sample")[0]).write_text("tampered")
        with self.assertRaisesRegex(ValueError, "raw file changed"):
            p.reuse_base(prep, baseline, _fixture=True)

    def test_cache_rejects_modified_sample_even_if_raw_origin_is_retained(self):
        prep, baseline, original = self.baseline_fixture()
        original["evaluations"]["base"]["records"][0]["completion_token_ids"] = [999]
        named.write_json(prep / "base/fragment.json", original)
        with self.assertRaisesRegex(ValueError, "actual original sample"):
            p.reuse_base(prep, baseline, _fixture=True)

    def test_actual_complete_history_is_required_and_zero_is_read_not_inserted(self):
        old_plan = p.checked_plan(named.HISTORY_PLAN)
        history = p.completed_history(named.HISTORY, old_plan)
        self.assertEqual(history["summary"]["real_curriculum_mixed_binary_optimizer_steps"], 0)
        self.assertEqual(history["per_stage_delta"], named.read(named.HISTORY / "delta.json"))
        with self.assertRaises(FileNotFoundError):
            p.completed_history(self.root / "incomplete_result", old_plan)
        wrong = copy.deepcopy(old_plan)
        wrong["plan_sha256"] = "other_round"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            p.completed_history(named.HISTORY, wrong)

    def test_dry_run_can_finish_missing_evidence_without_model_or_ready(self):
        with patch.object(p, "sample_challenger", side_effect=AssertionError("must not load Q")):
            report = p.main(self.root / "dry", preparation=self.root / "pending_prep",
                result=self.root / "pending_result", screen=self.root / "pending_screen", dry_run=True)
        self.assertEqual(report["status"], "pending_external_evidence")
        self.assertEqual(report["new_baseline_evaluation_rollouts"], 0)
        self.assertFalse((self.root / "dry/READY.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
