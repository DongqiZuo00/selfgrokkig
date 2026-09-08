"""Synthetic source/schema tests; actual CSV serialization is checked separately."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_verge_search_report import bundle_fixture
import verge_search_csv_contract as contract


class SearchCSVTests(unittest.TestCase):
    def test_six_tables_keep_full_matrix_and_budget(self):
        tables=contract.expected_tables(bundle_fixture())
        self.assertEqual([len(tables[k]) for k in contract.COLUMNS],[456,234,24,6,6,36])
        self.assertEqual(sum(r['loss_tokens'] for r in tables['generation_costs']),12582912)
        for name,columns in contract.COLUMNS.items():
            self.assertTrue(all(c in r for r in tables[name] for c in columns))

    def test_missing_is_not_zero_and_booleans_remain_booleans(self):
        tables=contract.expected_tables(bundle_fixture())
        event=tables['trajectory_events'][0]
        self.assertIsNone(event['first_event_update']);self.assertIs(event['event_observed'],False)
        self.assertIs(event['right_censored'],True);self.assertEqual(event['observation_loss_tokens'],524288)
        self.assertIsNone(tables['slurm_allocations'][0]['batch_max_rss_bytes'])

    def test_source_paths_keep_shared_and_worker_observations_distinct(self):
        rows=contract.expected_tables(bundle_fixture())['condition_profiles']
        for r in rows:
            p=r['source_artifact']
            if r['endpoint']=='initial':self.assertIn('/rounds/'+r['round_version']+'/initial/',p)
            elif r['endpoint']=='branch_endpoint':self.assertIn(f"_b{r['branch_index']}/branches/0/",p)
            else:self.assertTrue(p.endswith('/completion.jsonl'))

    def test_reject_incomplete_analysis_before_reading_csv(self):
        bundle=bundle_fixture();bundle['analysis']['endpoint_rows'].pop()
        with self.assertRaises(ValueError):contract.validate_csv(bundle,Path('nonexistent'))

    def test_reject_changed_provenance_before_reading_csv(self):
        bundle=bundle_fixture();receipt={'protocol':'verge_search_v1_csv_export','synthetic':False,
            'search_block_complete':False,'suite_complete':False}
        with patch.object(Path,'read_text',return_value=contract.json.dumps(receipt)):
            with self.assertRaisesRegex(RuntimeError,'provenance'):contract.validate_csv(bundle,Path('nonexistent'))

    def test_generation_count_missing_is_not_silently_filled(self):
        bundle=bundle_fixture();del bundle['analysis']['generation_ledger'][0]['rollouts']
        with self.assertRaises(ValueError):contract.expected_tables(bundle)


if __name__=='__main__':unittest.main()
