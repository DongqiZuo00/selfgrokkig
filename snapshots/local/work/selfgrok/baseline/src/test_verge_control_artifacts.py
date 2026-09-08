"""Synthetic serialization and completion gates; no real result marking."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_control_artifacts as artifacts
from verge_control_controller import write


class ControlArtifactTests(unittest.TestCase):
    def test_zero_missing_and_boolean_are_distinct(self):
        self.assertTrue(artifacts.same_cell('',None))
        self.assertFalse(artifacts.same_cell('',0))
        self.assertFalse(artifacts.same_cell('0',None))
        self.assertTrue(artifacts.same_cell('false',False))
        self.assertFalse(artifacts.same_cell('0',False))
        self.assertTrue(artifacts.same_cell('0.001953125',1/512))
        self.assertFalse(artifacts.same_cell('0.002',1/512))
        self.assertFalse(artifacts.same_cell('NaN',0.))
        self.assertTrue(artifacts.same_cell('9007199254740993',9007199254740993))
        self.assertFalse(artifacts.same_cell('9007199254740992',9007199254740993))

    def fixture(self, directory):
        expected = {name:[{column:'literal' for column in columns}] for name,columns in artifacts.COLUMNS.items()}
        expected['condition_profiles'][0]['observed_rate'] = 1/512
        expected['condition_profiles'][0]['rollouts'] = 512
        expected['trajectory_events'][0]['first_success_update'] = None
        receipt = {'protocol':'verge_control_v1_csv_export','synthetic':True,'suite_complete':False,'tables':[]}
        for name,columns in artifacts.COLUMNS.items():
            # Tiny RFC fixtures exercise the reader; these are not authored
            # result workbooks and are removed with the temporary test directory.
            with (directory/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow([expected[name][0][c] for c in columns])
            receipt['tables'].append({'name':name,'rows':1,'columns':len(columns),
                                     'typed_values_unchanged':True,'preview_rendered':True})
        write(directory/'csv_export_receipt.json',receipt)
        return expected

    def test_each_csv_cell_reconciles_to_source(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            expected = self.fixture(directory)
            with patch.object(artifacts,'expected_tables',return_value=expected):
                self.assertEqual(artifacts.validate_csv({'synthetic':True},directory)['csv_tables_verified'],5)
                path = directory/'condition_profiles.csv'
                path.write_text(path.read_text().replace('0.001953125','0.002'))
                with self.assertRaises(RuntimeError):
                    artifacts.validate_csv({'synthetic':True},directory)

    def test_missing_table_header_or_extra_row_is_rejected(self):
        for problem in ('header','extra','receipt'):
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                expected = self.fixture(directory)
                path = directory/'condition_profiles.csv'
                if problem == 'header':
                    path.write_text(path.read_text().replace('observed_rate','other_rate'))
                elif problem == 'extra':
                    path.write_text(path.read_text()+'extra\n')
                else:
                    receipt = artifacts.ctl.read(directory/'csv_export_receipt.json')
                    receipt['tables'].pop()
                    write(directory/'csv_export_receipt.json',receipt)
                with patch.object(artifacts,'expected_tables',return_value=expected):
                    with self.assertRaises(RuntimeError):
                        artifacts.validate_csv({'synthetic':True},directory)

    def test_synthetic_or_stale_bundle_cannot_finalize_real_control_block(self):
        data,snapshot,hardware = {},{'allocations':[]},{}
        for bundle in ({'synthetic':True}, {'synthetic':False,'analysis':{'stale':True}}):
            with patch.object(artifacts.report,'collect',return_value=(data,snapshot,hardware)), patch.object(artifacts.ctl,'read',return_value=bundle):
                with self.assertRaises(RuntimeError):
                    artifacts.validate()

    def test_completion_preserves_scope_to_only_three_controls(self):
        source = Path(artifacts.__file__).read_text()
        self.assertIn('suite_complete=False',source)
        self.assertIn('required_matrix_remaining=True',source)
        self.assertIn("review.get('synthetic') is not False",source)


if __name__ == '__main__':
    unittest.main()
