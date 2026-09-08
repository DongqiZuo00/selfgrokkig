"""CPU catalogue contracts and real-parser oracle checks when --parser is set."""
import argparse
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import stage_catalogue as c

PARSER_PATH = None


class CatalogueTests(unittest.TestCase):
    def test_nine_candidates_two_prompt_variants_share_identical_tasks(self):
        concise, official = c.build_rows("concise"), c.build_rows("official")
        self.assertEqual(len(concise), 9)
        for a, b in zip(concise, official):
            self.assertEqual(a["candidate_id"], b["candidate_id"])
            self.assertEqual(a["ground_truth"], b["ground_truth"])
            self.assertEqual(a["instance_tuple"], b["instance_tuple"])
            self.assertEqual(a["task_definition"], b["task_definition"])
            self.assertNotEqual(a["messages"], b["messages"])
            self.assertEqual(len(c.make_stage_rows(a["candidate_id"], "concise")), 1)

    def test_every_stage_nonconstant_binary_full_pass_and_fixed_expectations(self):
        for row in c.build_rows():
            definition = c.DEFINITION_BY_ID[row["candidate_id"]]
            self.assertGreaterEqual(len({case["expected_output"] for case in row["ground_truth"]}), 2)
            self.assertEqual(row["solver_reward"], "binary_full_pass")
            self.assertIsNone(row["hint"])
            for case in row["ground_truth"]:
                self.assertEqual(case["expected_output"], c.expectation(definition, case["input"]))
                self.assertTrue(case["expected_accepted"] and case["check_output"])

    def test_all_inputs_train_partition_and_no_protected_input_overlap(self):
        protected = set()
        for split in ("selection", "heldout", "test"):
            for line in (c.RELEASE / "benchmark/generated" / f"target_{split}.jsonl").read_text(encoding="utf-8").splitlines():
                protected.update(case["input"] for case in json.loads(line)["ground_truth"])
        for row in c.build_rows():
            inputs = [case["input"] for case in row["ground_truth"]]
            self.assertFalse(set(inputs) & protected)
            self.assertTrue(all(c.TARGET.split_of(x) == "train" for x in inputs))

    def test_instance_identity_is_two_sha_and_not_split_or_prompt_label(self):
        for row in c.build_rows():
            inputs = [case["input"] for case in row["ground_truth"]]
            expected = c.STAGES.instance_tuple_from_manifest(row["task_definition"], inputs)
            self.assertEqual(row["instance_tuple"], list(expected))
            self.assertEqual([len(x) for x in expected], [64, 64])
            self.assertEqual(expected, c.STAGES.instance_tuple_from_manifest(row["task_definition"], list(reversed(inputs))))
        with self.assertRaises(ValueError):
            c.STAGES.instance_tuple_from_manifest({"split": "train", "operation": "identity"}, ["B", "R"])

    def test_small2_omission_is_explicit_and_small3_covers_mutation(self):
        short = c.make_stage_rows("target_small2")[0]
        self.assertEqual([x["input"] for x in short["ground_truth"]], ["B", "R", "BB", "BR", "RR"])
        self.assertFalse(any("RB" in x["input"] for x in short["ground_truth"]))
        larger = c.make_stage_rows("target_small3")[0]
        self.assertTrue(any("RB" in x["input"] for x in larger["ground_truth"]))

    def test_model_rows_and_prompt_do_not_contain_reference_solution(self):
        for variant in c.VARIANTS:
            for row in c.build_rows(variant):
                serialized = json.dumps(row)
                oracle = c.reference_program(row["candidate_id"]).strip()
                self.assertNotIn(oracle, row["messages"][0]["content"])
                self.assertNotIn("reference_program", serialized)
                self.assertNotIn("NEXT marker", serialized)
                self.assertNotIn("SYNTHETIC_ORACLE", serialized)

    def test_all_candidate_specs_pass_four_applicable_validator_checks(self):
        protected = [c.STAGES.instance_tuple_from_manifest({"parameters": c.TARGET.PARAMETERS}, ["RB", ""])]
        registry = c.registered_families()
        for definition in c.DEFINITIONS:
            result = c.STAGES.validate_stage(c.stage_spec(definition["id"]), registry=registry, protected_tuples=protected)
            self.assertTrue(result.accepted, result.details)
            self.assertEqual(result.checks["hint_regret"], "NA")
            self.assertEqual(len(result.checks), 4)
            if definition["kind"] == "tape_transform":
                expr = c.expression(definition)
                for tape in c.stage_inputs(definition):
                    self.assertEqual(c.STAGES.evaluate_transform(expr, tape), c.expectation(definition, tape))

    def test_target_function_and_four_symbol_dsl_are_preserved(self):
        self.assertEqual(c.TARGET.PARAMETERS, {"prepend_sequence": "RBYGRB", "mutation_type": "replace_pattern_to_pattern", "source_pattern": "RB", "target_pattern": "YG"})
        for stage in ("target_small2", "target_small3"):
            d = c.DEFINITION_BY_ID[stage]
            for n in range(d["max_length"] + 1):
                for chars in itertools.product("RB", repeat=n):
                    tape = "".join(chars)
                    self.assertEqual(c.expectation(d, tape), "RBYGRB" + tape.replace("RB", "YG"))

    def test_stage_verifier_counts_exception_as_single_failure_and_preserves_raw(self):
        row = c.make_stage_rows("identity")[0]
        bad_input = row["ground_truth"][0]["input"]
        seen = []
        raw = "Text\n```manufactoria\n" + c.reference_program("identity") + "```"
        def make(program):
            seen.append(program)
            def run(tape):
                if tape == bad_input:
                    raise RuntimeError("fixture timeout")
                return SimpleNamespace(finished=True, final_tape=tape, rejection_reason=None, path=["start", "end"])
            return SimpleNamespace(process_robot=run)
        result = c.evaluate_stage(raw, row, make)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["passed_cases"], result["total_cases"] - 1)
        self.assertEqual(seen, [c.reference_program("identity").strip()])
        self.assertTrue(raw.startswith("Text"))

    def test_tampered_expected_output_or_identity_is_rejected(self):
        for change in ("output", "identity", "definition"):
            row = copy.deepcopy(c.make_stage_rows("append_R")[0])
            if change == "output":
                row["ground_truth"][0]["expected_output"] = ""
            elif change == "identity":
                row["instance_tuple"] = ["0" * 64, "1" * 64]
            else:
                row["task_definition"]["literal"] = "B"
            with self.assertRaises(ValueError):
                c.evaluate_stage("irrelevant", row, lambda _: None)

    def test_real_parser_all_references_and_wrong_constant_negative_control(self):
        if PARSER_PATH is None:
            self.skipTest("actual vendor parser supplied in remote CPU verification")
        parser = c.load_module(PARSER_PATH, "catalogue_unit_original_parser")
        constant = "START start:\n    NEXT consume\nPULLER_RB consume:\n    [R] consume\n    [B] consume\n    [EMPTY] end\nEND end"
        for row in c.build_rows():
            oracle = c.reference_program(row["candidate_id"])
            self.assertEqual(c.evaluate_stage("Text\n```manufactoria\n" + oracle + "```", row, parser.create_robot_factory)["reward"], 1)
            self.assertEqual(c.evaluate_stage(constant, row, parser.create_robot_factory)["reward"], 0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--parser", type=Path)
    args, rest = cli.parse_known_args()
    PARSER_PATH = args.parser
    unittest.main(argv=[__file__] + rest, verbosity=2)
