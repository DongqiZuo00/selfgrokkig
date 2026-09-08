"""CPU tests; injectable real vendor parser supplied by --parser.

Synthetic failing factories test bookkeeping only, not model quality.
"""
import argparse
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
                self.assertEqual(len({c["input"] for c in cases}), 24)
                for case in cases:
                    self.assertEqual(b.split_of(case["input"]), split)

    def test_split_tapes_are_disjoint_and_empty_is_covered(self):
        tapes = {split: {c["input"] for r in rows for c in r["ground_truth"]} for split, rows in self.rows.items()}
        for a, c in itertools.combinations(tapes, 2):
            self.assertFalse(tapes[a] & tapes[c])
        self.assertIn("", tapes[b.split_of("")])

    def test_binary_reward_and_class_minimum_synthetic(self):
        # One failing class must dominate f even though 20/24 tests pass overall.
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
        self.assertEqual(sum(x["pass"] for x in result["per_test"]), 20)

    def test_bad_parse_preserves_all_denominators(self):
        def fail(_):
            raise ValueError("synthetic parse rejection")
        result = b.evaluate_program("bad", self.cases, fail)
        self.assertFalse(result["parse_valid"])
        self.assertEqual(len(result["per_test"]), 24)
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
        self.assertEqual([x["pass"] for x in result["per_test"]], [0, 0] + [1] * 22)
        self.assertEqual(result["class_rates"][b.CLASS_IDS[0]], .5)

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
