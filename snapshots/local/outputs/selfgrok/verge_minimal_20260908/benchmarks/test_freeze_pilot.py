"""CPU checks for an unchanged target behind a compact pilot prompt view."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import freeze_pilot as f


class PilotViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.metadata = f.build_view()
        cls.stages = f.stage_validation_evidence()

    def test_original_ids_fields_and_ground_truth_exactly_preserved(self):
        for split, count in (("train", 128), ("selection", 8)):
            originals = f.read_jsonl(f.c.RELEASE / "benchmark/generated" / f"target_{split}.jsonl")[:count]
            self.assertEqual(len(self.rows[split]), count)
            for original, actual in zip(originals, self.rows[split]):
                self.assertEqual(actual["id"], original["id"])
                self.assertEqual({k: v for k, v in actual.items() if k != "messages"},
                                 {k: v for k, v in original.items() if k != "messages"})
                self.assertNotEqual(actual["messages"], original["messages"])

    def test_six_classes_and_36_tests_remain_on_every_instance(self):
        for split in ("train", "selection"):
            for row in self.rows[split]:
                frozen = f.contract(row)
                self.assertEqual(len(frozen["test_ids"]), 36)
                self.assertEqual(set(frozen["expected_classes"]), set(f.c.TARGET.CLASS_IDS))
        self.assertEqual(len(self.metadata["selection"]["test_contracts"]), 8)

    def test_hash_payload_and_round_request_contract_match(self):
        self.assertEqual(f.sha(self.metadata["prompt_view_hash_payload"]), self.rows["prompt_view_sha256"])
        selection = self.metadata["selection"]
        self.assertEqual(selection["prompt_view_sha256"], self.rows["prompt_view_sha256"])
        self.assertEqual(selection["manifest_id"], self.rows["manifest_id"])
        self.assertEqual(selection["instance_ids"], [r["id"] for r in self.rows["selection"]])
        self.assertEqual(len(selection["sample_ids"]), 32)
        self.assertEqual(len(set(selection["sample_ids"])), 32)
        # The first eight sample slots are a literal prefix for stage/final P8.
        self.assertEqual(selection["sample_ids"][:8], [f"rollout_{i:02d}" for i in range(8)])

    def test_one_unhinted_prompt_no_program_oracle(self):
        messages = self.rows["train"][0]["messages"]
        for split in ("train", "selection"):
            for row in self.rows[split]:
                self.assertEqual(row["messages"], messages)
                self.assertIsNone(row["hint"])
        content = messages[0]["content"]
        self.assertIn("length from 0 to 64", content)
        self.assertIn("Replace every nonoverlapping occurrence of RB with YG, then prepend RBYGRB", content)
        self.assertNotIn(f.c.TARGET.reference_program().strip(), content)
        self.assertNotIn("VERGE_TRAIN_HINT", content)

    def test_stage_evidence_binds_actual_cpu_tests_and_exact_specs(self):
        self.assertEqual(len(self.stages["stage_validations"]), 9)
        for sid, value in self.stages["stage_validations"].items():
            self.assertTrue(value["accepted"])
            self.assertFalse(value["synthetic_fixture"])
            self.assertEqual(value["checks"]["hint_regret"], "NA")
            self.assertEqual(value["spec_sha256"], f.sha(self.stages["specifications"][sid]))
            self.assertTrue(value["details"]["cpu_reference_evidence"]["all_exact_pass"])
            self.assertEqual(value["details"]["protected_actual_input_overlap"], 0)
        self.assertFalse(self.stages["hinted_target_certified_by_this_file"])

    def test_tampered_test_class_or_target_is_rejected(self):
        for change in ("class", "output", "missing"):
            row = copy.deepcopy(self.rows["selection"][0])
            if change == "class":
                row["ground_truth"][0]["test_class"] = "made_up"
            elif change == "output":
                row["ground_truth"][0]["expected_output"] = "WRONG"
            else:
                row["ground_truth"].pop()
            with self.assertRaises(ValueError):
                f.contract(row)

    def test_frozen_files_match_in_memory_and_round_sha(self):
        path = f.HERE / "generated/pilot_view_v1"
        stored_rows = json.loads((path / "target_rows.json").read_text(encoding="utf-8"))
        stored_meta = json.loads((path / "target_view_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(stored_rows, self.rows)
        self.assertEqual(stored_meta["target_rows_file_sha256"], f.file_sha(path / "target_rows.json"))
        module_path = f.HERE.parent / "round/round_cli.py"
        spec = importlib.util.spec_from_file_location("pilot_test_round_cli", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.sha(self.metadata["prompt_view_hash_payload"]), self.rows["prompt_view_sha256"])
        for sid, value in self.stages["stage_validations"].items():
            self.assertEqual(module.sha(self.stages["specifications"][sid]), value["spec_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
