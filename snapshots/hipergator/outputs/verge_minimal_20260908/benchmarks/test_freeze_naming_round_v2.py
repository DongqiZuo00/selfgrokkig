import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import freeze_naming_round_v2 as view


class NamingRoundViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.metadata, cls.catalogue, cls.evidence = view.build()
        cls.delta = view.read(view.NAMES)["prompt_delta"]
        cls.original_rows = view.read(view.V1 / "target_rows.json")
        cls.original_metadata = view.read(view.V1 / "target_view_metadata.json")

    def test_all_target_fields_tests_classes_and_ids_preserved(self):
        for split, count in (("train", 128), ("selection", 8)):
            self.assertEqual(len(self.rows[split]), count)
            for old, new in zip(self.original_rows[split], self.rows[split]):
                self.assertEqual({k: v for k, v in new.items() if k != "messages"},
                                 {k: v for k, v in old.items() if k != "messages"})
                self.assertEqual(new["messages"][0]["content"], old["messages"][0]["content"] + "\n\n" + self.delta)
                view.pilot.contract(new)
        for field in ("instance_ids", "sample_ids", "test_contracts"):
            self.assertEqual(self.metadata["selection"][field], self.original_metadata["selection"][field])
        self.assertEqual(self.rows["source_train_manifest_sha256"], self.original_rows["source_train_manifest_sha256"])

    def test_catalogue_and_reused_real_validation_preserved(self):
        old = view.read(view.HERE / "generated/catalogue.json")
        old_evidence = view.read(view.V1 / "stage_validation_evidence.json")
        for a, b in zip(old["stages"], self.catalogue["stages"]):
            for key in ("stage_id", "kind", "spec"):
                self.assertEqual(a[key], b[key])
            pairs = list(zip(a["rows"], b["rows"]))
            for variant in a["rows_by_variant"]:
                pairs += list(zip(a["rows_by_variant"][variant], b["rows_by_variant"][variant]))
            for prior, actual in pairs:
                self.assertEqual(view.changed_row(prior, self.delta), actual)
            evidence = copy.deepcopy(self.evidence["stage_validations"][a["stage_id"]])
            evidence["details"].pop("prompt_view_binding")
            self.assertEqual(evidence, old_evidence["stage_validations"][a["stage_id"]])
        self.assertEqual(self.evidence["specifications"], old_evidence["specifications"])
        self.assertFalse(self.evidence["prompt_treatment_model_effectiveness_established"])

    def test_new_hashes_and_execute_round_real_input_loader(self):
        self.assertNotEqual(self.rows["manifest_id"], self.original_rows["manifest_id"])
        self.assertNotEqual(self.rows["prompt_view_sha256"], self.original_rows["prompt_view_sha256"])
        self.assertEqual(view.pilot.sha(self.metadata["prompt_view_hash_payload"]), self.rows["prompt_view_sha256"])
        self.assertEqual(view.pilot.sha(self.metadata["stage_prompt_view_hash_payload"]), self.catalogue["prompt_view_sha256"])
        path = view.HERE.parent / "runtime/execute_round.py"
        spec = importlib.util.spec_from_file_location("naming_round_view_read_test", path)
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        # CPU-only read contract. This is not a Challenger proposal or run plan.
        chosen = [{k: s[k] for k in ("stage_id", "kind", "spec")} for s in self.catalogue["stages"]]
        read_contract = {"selection": self.metadata["selection"],
            "proposal": {"curricula": {"g1": chosen[:3], "g2": chosen[3:6], "g3": chosen[6:]}}}
        with tempfile.TemporaryDirectory(dir=view.HERE) as tmp:
            view.freeze(Path(tmp))
            rows, stages = runtime.load_inputs(read_contract, Path(tmp) / "target_rows.json", Path(tmp) / "catalogue.json")
            self.assertEqual(rows, self.rows)
            self.assertEqual(len(stages), 9)
            bad = copy.deepcopy(read_contract)
            bad["selection"]["prompt_view_sha256"] = self.original_rows["prompt_view_sha256"]
            with self.assertRaisesRegex(ValueError, "prompt view"):
                runtime.load_inputs(bad, Path(tmp) / "target_rows.json", Path(tmp) / "catalogue.json")

    def test_immutable_freeze_and_v1_source_preservation(self):
        sources = [view.V1 / n for n in ("target_rows.json", "target_view_metadata.json", "stage_validation_evidence.json")]
        sources += [view.HERE / "generated/catalogue.json", view.NAMES]
        before = {str(p): view.pilot.file_sha(p) for p in sources}
        with tempfile.TemporaryDirectory(dir=view.HERE) as tmp:
            output = Path(tmp)
            view.freeze(output)
            hashes = {p.name: view.pilot.file_sha(p) for p in output.iterdir()}
            view.freeze(output)
            self.assertEqual(hashes, {p.name: view.pilot.file_sha(p) for p in output.iterdir()})
            (output / "catalogue.json").write_text("changed", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                view.freeze(output)
        self.assertEqual(before, {str(p): view.pilot.file_sha(p) for p in sources})


if __name__ == "__main__":
    unittest.main(verbosity=2)
