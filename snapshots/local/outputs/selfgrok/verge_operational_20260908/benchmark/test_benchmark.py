"""CPU tests; injectable real vendor parser supplied by --parser.

Synthetic failing factories test bookkeeping only, not model quality.
"""
import argparse
import ast
from collections import Counter
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import benchmark as b

ROOT = Path(__file__).resolve().parent / "generated"
PARSER_PATH = None


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {s: [json.loads(x) for x in (ROOT / f"target_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for s in b.SPLIT_COUNTS}
        cls.cases = cls.rows["train"][0]["ground_truth"]

    def test_class_rules_and_empty(self):
        examples = {"": "matches_0__pending_r_0", "R": "matches_0__pending_r_1",
                    "RB": "matches_1__pending_r_0", "RBR": "matches_1__pending_r_1",
                    "RBRB": "matches_2plus__pending_r_0", "RBRBR": "matches_2plus__pending_r_1"}
        for tape, cls in examples.items():
            self.assertEqual(b.test_class(tape), cls)
        for invalid in ("Y", "RGB", "R" * 65, None):
            with self.assertRaises(ValueError):
                b.test_class(invalid)

    def test_every_suite_has_fixed_complete_disjoint_classes(self):
        for split, rows in self.rows.items():
            self.assertEqual(len(rows), b.SPLIT_COUNTS[split])
            for row in rows:
                self.assertIsNone(row["hint"])
                cases = row["ground_truth"]
                self.assertEqual(Counter(c["test_class"] for c in cases), Counter(dict.fromkeys(b.CLASS_IDS, b.PER_CLASS)))
                self.assertEqual(len({c["input"] for c in cases}), b.PER_CLASS * len(b.CLASS_IDS))
                for case in cases:
                    self.assertEqual(b.split_of(case["input"]), split)

    def test_split_tapes_are_disjoint_and_empty_is_covered(self):
        tapes = {split: {c["input"] for r in rows for c in r["ground_truth"]} for split, rows in self.rows.items()}
        for a, c in itertools.combinations(tapes, 2):
            self.assertFalse(tapes[a] & tapes[c])
        self.assertIn("", tapes[b.split_of("")])

    def test_binary_reward_and_class_minimum_synthetic(self):
        # One failing class must dominate f even though 30/36 tests pass overall.
        failed_class = b.CLASS_IDS[-1]
        def make(_):
            def process(x):
                output = b.expected_output(x)
                return SimpleNamespace(finished=True, final_tape=output if b.test_class(x) != failed_class else "WRONG",
                                       path=["start", "end"], rejection_reason=None)
            return SimpleNamespace(process_robot=process)
        result = b.evaluate_program("synthetic", self.cases, make)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["f"], 0)
        self.assertEqual(sum(x["pass"] for x in result["per_test"]), b.PER_CLASS * (len(b.CLASS_IDS) - 1))

    def test_bad_parse_preserves_all_denominators(self):
        def fail(_):
            raise ValueError("synthetic parse rejection")
        result = b.evaluate_program("bad", self.cases, fail)
        self.assertFalse(result["parse_valid"])
        self.assertEqual(len(result["per_test"]), b.PER_CLASS * len(b.CLASS_IDS))
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["f"], 0)

    def test_timeout_and_exception_do_not_erase_other_tests(self):
        timeout_tape, exception_tape = [c["input"] for c in self.cases[:2]]
        def make(_):
            def process(x):
                if x == exception_tape:
                    raise RuntimeError("synthetic runtime error")
                return SimpleNamespace(finished=x != timeout_tape, final_tape=b.expected_output(x),
                    path=["scan"] * (1000 if x == timeout_tape else 2),
                    rejection_reason="Maximum iterations exceeded" if x == timeout_tape else None)
            return SimpleNamespace(process_robot=process)
        result = b.evaluate_program("synthetic", self.cases, make)
        self.assertEqual([x["pass"] for x in result["per_test"]], [0, 0] + [1] * (b.PER_CLASS * len(b.CLASS_IDS) - 2))
        self.assertEqual(result["class_rates"][b.CLASS_IDS[0]], (b.PER_CLASS - 2) / b.PER_CLASS)

    def test_original_four_fraction_thresholds_are_distinct_events(self):
        thresholds = [1 / (b.PER_CLASS * len(b.CLASS_IDS)), .25, .5, .75]
        # With the abandoned four-case draft, first two events were identical.
        minimum_success_counts = [next(n for n in range(b.PER_CLASS + 1) if n / b.PER_CLASS >= t) for t in thresholds]
        self.assertEqual(minimum_success_counts, [1, 2, 3, 5])
        self.assertEqual(len(set(minimum_success_counts)), len(thresholds))

    def test_missing_duplicate_and_changed_metadata_rejected(self):
        for cases in ([], self.cases[:-1], [self.cases[1]] + self.cases[1:],
                      [dict(self.cases[0], expected_output="")] + self.cases[1:]):
            with self.assertRaises(ValueError):
                b.evaluate_program("unused", cases, lambda _: None)

    def test_reference_not_present_in_target_prompts(self):
        for rows in self.rows.values():
            for row in rows:
                content = row["messages"][0]["content"]
                self.assertNotIn("delimiter", content)
                self.assertNotIn("pending_r", content)
                self.assertNotIn(b.reference_program(), content)

    def test_extraction_function_matches_original_project_ast(self):
        original_path = Path(__file__).resolve().parents[2] / "verge_review_20260908/review_copy/src/common.py"
        original_tree = ast.parse(original_path.read_text(encoding="utf-8"))
        actual_tree = ast.parse(Path(b.__file__).read_text(encoding="utf-8"))
        original = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_program")
        actual = next(n for n in actual_tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_program")
        self.assertEqual(ast.dump(original), ast.dump(actual))

    def test_complete_fenced_and_prose_reference_extract_without_raw_mutation(self):
        reference = b.reference_program()
        raw = "Some reasoning.\n```manufactoria\n" + reference + "```\nExplanation after code."
        original = raw
        self.assertEqual(b.extract_program(raw), reference.strip())
        self.assertEqual(raw, original)
        seen = []
        def capture(program):
            seen.append(program)
            def process(x):
                return SimpleNamespace(finished=True, final_tape=b.expected_output(x), path=["start", "end"], rejection_reason=None)
            return SimpleNamespace(process_robot=process)
        self.assertEqual(b.evaluate_program(raw, self.cases, capture)["reward"], 1)
        self.assertEqual(seen, [reference.strip()])

    def test_truncated_block_and_complete_unclosed_fence_follow_original_fallback(self):
        truncated = "Explanation\n```manufactoria\nSTART start:\n    NEXT missing\nPULLER_RB missing:"
        self.assertEqual(b.extract_program(truncated), truncated)
        complete_unclosed = "Explanation\n```manufactoria\n" + b.reference_program()
        self.assertEqual(b.extract_program(complete_unclosed), b.reference_program().strip())

    def test_extraction_keeps_first_invalid_complete_block(self):
        invalid = "START start:\n    NEXT nonexistent\nEND end"
        raw = "Reasoning\n```manufactoria\n" + invalid + "\n```\n" + "```manufactoria\n" + b.reference_program() + "```"
        self.assertEqual(b.extract_program(raw), invalid)

    def test_real_parser_completion_extraction_and_no_reward_selection(self):
        if PARSER_PATH is None:
            self.skipTest("real vendor parser supplied on remote CPU run")
        parser = b.load_python(PARSER_PATH, "verge_completion_real_vendor")
        ref = b.reference_program()
        complete = "Reasoning\n```manufactoria\n" + ref + "```\nExplanation"
        self.assertEqual(b.evaluate_program(complete, self.cases, parser.create_robot_factory)["reward"], 1)
        self.assertEqual(b.evaluate_program("Unclosed\n```manufactoria\n" + ref, self.cases, parser.create_robot_factory)["reward"], 1)
        truncated = "Unclosed\n```manufactoria\nSTART start:\n    NEXT end\nEND"
        self.assertFalse(b.evaluate_program(truncated, self.cases, parser.create_robot_factory)["parse_valid"])
        invalid = "START start:\n    NEXT nonexistent\nEND end"
        two_blocks = "```manufactoria\n" + invalid + "\n```\n```manufactoria\n" + ref + "```"
        result = b.evaluate_program(two_blocks, self.cases, parser.create_robot_factory)
        self.assertFalse(result["parse_valid"])
        self.assertEqual(result["reward"], 0)

    def test_actual_vendor_parser_reference_and_timeout(self):
        if PARSER_PATH is None:
            self.skipTest("real vendor parser only supplied on remote CPU run")
        parser = b.load_python(PARSER_PATH, "verge_test_actual_vendor")
        result = b.evaluate_program(b.reference_program(), self.cases, parser.create_robot_factory)
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["f"], 1)
        loop = "START start:\n    NEXT loop\nPAINTER_RED loop:\n    NEXT loop\nEND end"
        result = b.evaluate_program(loop, self.cases, parser.create_robot_factory)
        self.assertTrue(result["parse_valid"])
        self.assertEqual(result["reward"], 0)
        self.assertTrue(all(x["steps"] == 1000 for x in result["per_test"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "generated")
    cli.add_argument("--parser", type=Path)
    args, remaining = cli.parse_known_args()
    ROOT, PARSER_PATH = args.data, args.parser
    unittest.main(argv=[__file__] + remaining, verbosity=2)
