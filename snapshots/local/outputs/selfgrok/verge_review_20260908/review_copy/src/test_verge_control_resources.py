"""Synthetic Slurm records only; no actual accounting queries or new jobs."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_control_resources as resources
from verge_control_controller import write, LABELS
from verge_book_resources import parse_sacct, summarize


class ControlResourceTests(unittest.TestCase):
    def registry(self, root, **updates):
        record = {"round":0,"labels":list(LABELS),"workers":"100","coordinator":"101"}
        record.update(updates)
        write(root / "jobs/round_00.json",record)
        return resources.registry(root)

    def test_three_workers_and_no_parent_batch_double_counting(self):
        with tempfile.TemporaryDirectory() as temp:
            registered,parents = self.registry(Path(temp))
            self.assertEqual(set(registered),{"100_0","100_1","100_2","101"})
            lines = ["100|COMPLETED|3600|8|gres/gpu=1,mem=32G|", "101|COMPLETED|60|2|mem=4G|"]
            for i in range(3):
                lines += [f"100_{i}|COMPLETED|3600|8|gres/gpu=1,gres/gpu:b200=1,mem=32G|",
                          f"100_{i}.batch|COMPLETED|3600|8|gres/gpu=1,mem=32G|1024K"]
            result = summarize(parse_sacct("\n".join(lines)),registered)
            self.assertEqual(result["gpu_allocation_hours"],3.)
            self.assertAlmostEqual(result["cpu_allocation_core_hours"],24+2/60)
            self.assertTrue(result["accounting_complete"])
            self.assertEqual(result["allocations"][0]["batch_max_rss_bytes"],1048576)

    def test_previous_failed_allocations_remain_in_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            registered,parents = self.registry(Path(temp),historical_workers=["99"])
            self.assertEqual(parents,["99","100","101"])
            self.assertEqual(len(registered),7)
            self.assertIn("99_2",registered)

    def test_duplicated_or_foreign_registry_rejected(self):
        for update in ({"historical_workers":["100"]},{"coordinator":"100"},{"workers":"bad"}):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(RuntimeError):
                    self.registry(Path(temp),**update)

    def test_no_jobs_does_not_mean_completed_experiment(self):
        with patch.object(resources,"registry",return_value=({},[])), patch.object(resources.subprocess,"run") as run:
            result = resources.collect()
            self.assertFalse(result["accounting_complete"])
            self.assertEqual(result["registered_parent_jobs"],0)
            self.assertFalse(result["suite_complete"])
            run.assert_not_called()

    def test_collect_queries_only_recorded_parents(self):
        with tempfile.TemporaryDirectory() as temp:
            registered,parents = self.registry(Path(temp))
            with patch.object(resources,"registry",return_value=(registered,parents)), patch.object(resources.subprocess,"run",return_value=SimpleNamespace(stdout="")) as run:
                result = resources.collect()
                self.assertIn("100,101",run.call_args.args[0])
                self.assertFalse(result["accounting_complete"])
                self.assertEqual(len(result["missing_individual_accounting_rows"]),4)


if __name__ == "__main__":
    unittest.main()
