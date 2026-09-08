"""Resource-accounting fixtures, no Slurm calls."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from verge_book_resources import memory_bytes, parse_sacct, summarize, collect, SUITE


class ResourceTests(unittest.TestCase):
    def test_memory_units_and_missing(self):
        self.assertIsNone(memory_bytes(""))
        self.assertEqual(memory_bytes("32G"), 32 * 1024**3)
        self.assertEqual(memory_bytes("32768Mn"), 32 * 1024**3)

    def test_no_parent_step_or_typed_gpu_double_count(self):
        lines = ["10|COMPLETED|3600|8|cpu=8,mem=32G,gres/gpu=1,gres/gpu:b200=1|",
                 "10.batch|COMPLETED|3600|8|cpu=8,gres/gpu=1|1048576K",
                 "10.extern|COMPLETED|3600|8|cpu=8,gres/gpu=1|",
                 "999|RUNNING|3600|16|cpu=16,gres/gpu=2|0K"]
        result = summarize(parse_sacct("\n".join(lines)), {"10": {"stage": "prepare"}})
        self.assertEqual(result["gpu_allocation_hours"], 1)
        self.assertEqual(result["cpu_allocation_core_hours"], 8)
        self.assertEqual(result["allocations"][0]["batch_max_rss_bytes"], 1024**3)

    def test_array_elements_counted_once(self):
        lines = [f"20_{i}|COMPLETED|900|8|cpu=8,mem=32G,gres/gpu=1,gres/gpu:b200=1|" for i in range(4)]
        result = summarize(parse_sacct("\n".join(lines)), {"20": {"stage": "branches"}})
        self.assertEqual(result["gpu_allocation_hours"], 1)
        self.assertTrue(result["accounting_complete"])

    def test_pending_range_not_mistaken_for_complete(self):
        result = summarize(parse_sacct("20_[0-3]|PENDING|0|0||"), {"20": {"stage": "branches"}})
        self.assertFalse(result["accounting_complete"])
        self.assertEqual(len(result["missing_individual_accounting_rows"]), 4)

    def test_failed_allocation_still_costs_resources(self):
        result = summarize(parse_sacct("30|FAILED|3600|8|cpu=8,mem=32G,gres/gpu=1|"), {"30": {"stage": "finish"}})
        self.assertEqual(result["gpu_allocation_hours"], 1)
        self.assertTrue(result["accounting_complete"])
        self.assertFalse(result["all_registered_jobs_successful"])

    def test_collect_uses_active_suite_and_registered_jobs_only(self):
        answer = SimpleNamespace(stdout="30|COMPLETED|3600|8|cpu=8,mem=32G,gres/gpu=1|\n")
        with patch("verge_book_resources.registry", return_value={"30": {"stage": "finish"}}), \
             patch("verge_book_resources.subprocess.run", return_value=answer) as command:
            result = collect()
        self.assertIn(f"raw_results/{SUITE}/jobs", result["scope"])
        self.assertEqual(result["gpu_allocation_hours"], 1)
        self.assertEqual(command.call_args[0][0][5], "30")


if __name__ == "__main__":
    unittest.main()
