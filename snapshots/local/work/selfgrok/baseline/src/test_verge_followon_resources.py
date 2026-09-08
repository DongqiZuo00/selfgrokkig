"""Synthetic Slurm allocation rows; no live accounting or model requests."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_followon_resources as resources
from verge_followon_controller import write
from verge_book_resources import parse_sacct,summarize


class ResourceTests(unittest.TestCase):
    def registry(self,root,**updates):
        record={'candidate_count':72,'indices':list(range(72)),'workers':'100','coordinator':'101'}
        record.update(updates);write(root/'jobs/block.json',record)
        return resources.registry(root)

    def test_72_workers_no_parent_or_batch_double_cost(self):
        with tempfile.TemporaryDirectory() as d:
            registered,parents=self.registry(Path(d));self.assertEqual(len(registered),73)
            lines=['100|COMPLETED|3600|8|gres/gpu=1,mem=32G|','101|COMPLETED|60|2|mem=4G|']
            for i in range(72):
                lines += [f'100_{i}|COMPLETED|3600|8|gres/gpu=1,gres/gpu:b200=1,mem=32G|',
                          f'100_{i}.batch|COMPLETED|3600|8|gres/gpu=1,mem=32G|1024K']
            result=summarize(parse_sacct('\n'.join(lines)),registered)
            self.assertEqual(result['gpu_allocation_hours'],72.)
            self.assertAlmostEqual(result['cpu_allocation_core_hours'],576+2/60)
            self.assertTrue(result['accounting_complete'])

    def test_historical_subset_recoveries_retained(self):
        with tempfile.TemporaryDirectory() as d:
            registered,parents=self.registry(Path(d),historical_workers=[{'job_id':'99','indices':[0,71]}],historical_coordinators=['98'])
            self.assertEqual(parents,['98','99','100','101']);self.assertEqual(len(registered),76)
            self.assertIn('99_71',registered);self.assertNotIn('99_70',registered)

    def test_duplicates_and_out_of_range_rejected(self):
        for changes in ({'coordinator':'100'},{'indices':[0]},{'historical_workers':[{'job_id':'99','indices':[72]}]},
                        {'historical_workers':[{'job_id':'100','indices':[1]}]}):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(RuntimeError):self.registry(Path(d),**changes)

    def test_current_recovery_subset_keeps_original_all_candidate_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            registered,parents=self.registry(Path(d),worker_indices=[1,70],
                historical_workers=[{'job_id':'99','indices':list(range(72))}])
            self.assertEqual(len(registered),75)
            self.assertIn('99_0',registered);self.assertIn('100_70',registered)
            self.assertNotIn('100_0',registered)

    def test_no_jobs_does_not_mean_complete(self):
        with patch.object(resources,'registry',return_value=({},[])),patch.object(resources.subprocess,'run') as run:
            result=resources.collect();self.assertFalse(result['accounting_complete']);run.assert_not_called()

    def test_only_registered_jobs_queried(self):
        with tempfile.TemporaryDirectory() as d:
            registered,parents=self.registry(Path(d))
            with patch.object(resources,'registry',return_value=(registered,parents)), \
                    patch.object(resources.subprocess,'run',return_value=SimpleNamespace(stdout='')) as run:
                result=resources.collect();self.assertIn('100,101',run.call_args.args[0])
                self.assertEqual(len(result['missing_individual_accounting_rows']),73)
                self.assertFalse(result['accounting_complete'])


if __name__=='__main__':unittest.main()
