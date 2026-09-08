"""Read-only CSV reconciliation tests using in-memory malformed fixtures."""
import copy
import csv
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import verge_followon_csv_contract as contract
from test_verge_followon_report import data_fixture,hardware_fixture


def bundle_fixture():
    data=data_fixture();data['synthetic']=True
    for c in data['generation_ledger']:
        c.update(nonzero_advantage_tokens=0,raw_rollouts=832,binary_verifier_test_calls=832,
                 source_endpoint_reused_without_generation=True,
                 uncommitted_failed_attempt_generation_included=False)
    snapshot,registered=hardware_fixture()
    for a in snapshot['allocations']:a['batch_max_rss_bytes']=None
    return {'synthetic':True,'analysis':data,'resources':snapshot}


def memory_files(bundle):
    """Reader test bytes only: no spreadsheet artifact is authored or modified."""
    tables=contract.expected_tables(bundle);files={};receipt={
        'protocol':'verge_followon_v1_csv_export','synthetic':True,
        'suite_complete':False,'followon_block_complete':False,'tables':[]}
    for name,columns in contract.COLUMNS.items():
        output=io.StringIO(newline='');writer=csv.writer(output);writer.writerow(columns)
        for row in tables[name]:
            writer.writerow(['' if row[k] is None else str(row[k]).lower() if isinstance(row[k],bool) else row[k] for k in columns])
        files[name+'.csv']=output.getvalue()
        receipt['tables'].append({'name':name,'rows':len(tables[name]),'columns':len(columns),
            'typed_values_unchanged':True,'preview_rendered':True,'preview_columns':8})
    return files,receipt


class FollowonCsvContractTests(unittest.TestCase):
    def setUp(self):
        self.bundle=bundle_fixture();self.files,self.receipt=memory_files(self.bundle)

    def check(self):
        files=self.files
        def opened(path,*args,**kwargs):return io.StringIO(files[path.name],newline='')
        with patch.object(contract,'read',return_value=self.receipt),patch.object(Path,'open',opened):
            return contract.validate_csv(self.bundle,Path('synthetic-memory-only'))

    def mutate(self,name,row,column,value):
        rows=list(csv.reader(io.StringIO(self.files[name+'.csv'])))
        rows[row][contract.COLUMNS[name].index(column)]=value
        text=io.StringIO(newline='');csv.writer(text).writerows(rows);self.files[name+'.csv']=text.getvalue()

    def test_all_seven_tables_and_all_cells(self):
        self.assertEqual(self.check()['csv_tables_verified'],7)
        self.assertEqual(len(contract.COLUMNS['trajectory_events']),33)
        self.assertFalse(self.check()['followon_block_complete'])

    def test_undefined_concordance_is_not_zero(self):
        self.mutate('cohort_rankings',1,'concordance','0')
        with self.assertRaisesRegex(ValueError,'value differs'):self.check()

    def test_false_is_not_numeric_zero(self):
        self.mutate('trajectory_events',1,'event_observed','0')
        with self.assertRaisesRegex(ValueError,'value differs'):self.check()

    def test_missing_source_event_has_no_fake_update(self):
        self.mutate('trajectory_events',1,'first_event_update','0')
        with self.assertRaisesRegex(ValueError,'value differs'):self.check()

    def test_source_path_is_independently_reconstructed(self):
        self.mutate('condition_profiles',1,'source_artifact','raw_results/foreign/target.jsonl')
        with self.assertRaisesRegex(ValueError,'value differs'):self.check()

    def test_missing_extra_rows_and_header_rejected(self):
        original=self.files['condition_profiles.csv']
        for content in ('\n'.join(original.splitlines()[:-1]),original+'extra\n',original.replace('observed_rate','wrong_rate',1)):
            self.files['condition_profiles.csv']=content
            with self.subTest(content=content[:30]),self.assertRaisesRegex(ValueError,'shape or columns'):self.check()

    def test_receipt_flags_dimensions_and_duplicates(self):
        original=copy.deepcopy(self.receipt)
        variants=[]
        for field in ('synthetic','followon_block_complete','suite_complete'):
            r=copy.deepcopy(original);r[field]=not r[field];variants.append(r)
        for field,value in (('rows',1),('columns',1),('preview_columns',7),('typed_values_unchanged',False),('preview_rendered',False)):
            r=copy.deepcopy(original);r['tables'][0][field]=value;variants.append(r)
        r=copy.deepcopy(original);r['tables'][0]=r['tables'][1];variants.append(r)
        for r in variants:
            self.receipt=r
            with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
