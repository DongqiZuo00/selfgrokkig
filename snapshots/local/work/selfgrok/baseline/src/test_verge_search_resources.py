"""Synthetic search resource bookkeeping; no Slurm queries or allocations."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_search_resources as resources
from verge_search_controller import write
from verge_book_resources import parse_sacct,summarize


class ResourceTests(unittest.TestCase):
    def fixture(self,root,**updates):
        record={'round':0,'version':'verge_search_v1_r00','branches':[0,1,2,3],
                'prepare':'99','workers':'100','finish':'101'}
        record.update(updates);write(root/'jobs/round_00.json',record)
        return resources.registry(root)

    def test_shared_evaluation_and_workers_counted_without_parent_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,parents=self.fixture(Path(tmp));self.assertEqual(len(registered),6)
            lines=[f'{j}|COMPLETED|3600|8|gres/gpu=1,gres/gpu:b200=1,mem=32G|' for j in ('99','100','101')]
            for i in range(4):lines += [f'100_{i}|COMPLETED|3600|8|gres/gpu=1,gres/gpu:b200=1,mem=32G|',
                                        f'100_{i}.batch|COMPLETED|3600|8|gres/gpu=1,mem=32G|1024K']
            result=summarize(parse_sacct('\n'.join(lines)),registered)
            self.assertEqual(result['gpu_allocation_hours'],6.)
            self.assertEqual(result['cpu_allocation_core_hours'],48.)
            self.assertTrue(result['accounting_complete'])

    def test_current_subset_recovery_keeps_all_original_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,parents=self.fixture(Path(tmp),historical_workers=[{'job_id':'90','indices':[0,1,2,3]}],
                worker_indices=[2],historical_prepare=['89'],historical_finish=['91'])
            self.assertEqual(parents,['89','90','91','99','100','101']);self.assertEqual(len(registered),9)
            self.assertIn('90_0',registered);self.assertIn('100_2',registered);self.assertNotIn('100_0',registered)

    def test_duplicate_or_out_of_range_registration_rejected(self):
        for changed in [{'finish':'100'},{'branches':[0,1]}, {'worker_indices':[4]},
                        {'historical_workers':[{'job_id':'100','indices':[2]}]}]:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):self.fixture(Path(tmp),**changed)

    def test_six_round_registry_has_36_allocations_not_18_parent_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for r in range(6):
                write(root/'jobs'/f'round_{r:02d}.json',{'round':r,'version':f'verge_search_v1_r{r:02d}',
                    'branches':[0,1,2,3],'prepare':str(100+r*3),'workers':str(101+r*3),'finish':str(102+r*3)})
            registry,parents=resources.registry(root);self.assertEqual(len(registry),36);self.assertEqual(len(parents),18)

    def test_failed_recovery_cost_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,_=self.fixture(Path(tmp),historical_workers=[{'job_id':'90','indices':[2]}])
            result=summarize(parse_sacct('90_2|FAILED|1800|8|gres/gpu=1,mem=32G|'),registered)
            self.assertEqual(result['gpu_allocation_hours'],.5);self.assertFalse(result['accounting_complete'])
            self.assertFalse(result['all_registered_jobs_successful'])

    def test_no_jobs_is_not_a_complete_comparison(self):
        with patch.object(resources,'registry',return_value=({},[])),patch.object(resources.subprocess,'run') as run:
            self.assertFalse(resources.collect()['accounting_complete']);run.assert_not_called()

    def test_only_registered_parents_queried(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered,parents=self.fixture(Path(tmp))
            with patch.object(resources,'registry',return_value=(registered,parents)),patch.object(resources.subprocess,'run',return_value=SimpleNamespace(stdout='')) as run:
                result=resources.collect();self.assertIn('99,100,101',run.call_args.args[0])
                self.assertEqual(len(result['missing_individual_accounting_rows']),6)


if __name__=='__main__':unittest.main()
