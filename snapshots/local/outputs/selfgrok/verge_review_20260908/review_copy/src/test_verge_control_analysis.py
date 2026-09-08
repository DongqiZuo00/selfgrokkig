"""Synthetic analysis contracts only; no pretrained-model draws or Slurm calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import verge_control_analysis as analysis
from verge_control_controller import write, LABELS
from verge_independent_sampling import PROTOCOL
from verge_round_core import condition_names


class ControlAnalysisTests(unittest.TestCase):
    @staticmethod
    def jsonl(path, records):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')

    def endpoint(self, directory, kind='target', *, n=2, draws=2):
        records = [{"instance_id":f"i{i//draws}","instance_index":i//draws,"rollout_index":i%draws,
                    "sampling_protocol":PROTOCOL,"sampling_parent_seed":100+(i//draws)*draws,
                    "sampling_child_seed":100+i,"prompt_token_ids":[1],"completion_token_ids":[2,3],
                    "completion_tokens":2,"max_tokens":2048,"passed_cases":0,"total_cases":2,"reward":0}
                   for i in range(n*draws)]
        request = {"prompt":[[1]]*n,"n":draws,"seed":100}
        response = {"backend_protocol":PROTOCOL,"prompt_parent_seeds":[100+i*draws for i in range(n)],
                    "child_seed_count":n*draws,"choices":[{"index":i} for i in range(n*draws)]}
        write(directory / (kind+'.response.json'),{"backend_protocol":PROTOCOL,"request":request,"response":response})
        self.jsonl(directory / (kind+'.jsonl'),records)
        if kind == 'target':
            vectors = {f"{r['instance_id']}::{r['rollout_index']}":[1,0,0,0,0,0,0,0,0,0,0] for r in records}
            summary = {"condition_names":condition_names(),"keyed_vectors":vectors,"counts":[n*draws]+[0]*10,
                       "rates":[1.]+[0.]*10,"rollouts":n*draws}
        else:
            summary = {"full_pass_rate":0.,"mean_case_fraction":0.}
        return records,summary

    def test_fixed_endpoint_profile_and_raw_full_success_remain_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _,summary = self.endpoint(directory)
            result = analysis.verified_endpoint(directory,'target',summary,2,2)
            self.assertEqual(result['matrix'].shape,(2,2,11))
            self.assertTrue((result['matrix'][:,:,0] == 1).all())
            self.assertTrue((result['matrix'][:,:,-1] == 0).all())

    def test_duplicate_keys_non_nested_or_stale_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records,summary = self.endpoint(directory)
            for transform in ('count','nested','reward'):
                bad = copy.deepcopy(summary)
                if transform == 'count':
                    bad['counts'][0] = 0
                elif transform == 'nested':
                    bad['keyed_vectors']['i0::0'][2] = 1
                else:
                    bad['keyed_vectors']['i0::0'] = [1]*11
                with self.assertRaises(RuntimeError):
                    analysis.verified_endpoint(directory,'target',bad,2,2)
            records[2]['instance_id'] = records[0]['instance_id']
            self.jsonl(directory / 'target.jsonl',records)
            with self.assertRaises(RuntimeError):
                analysis.verified_endpoint(directory,'target',summary,2,2)

    def test_foreign_stream_and_partial_reward_mislabel_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for field,value in (('sampling_child_seed',99),('reward',1),('max_tokens',4096)):
                records,summary = self.endpoint(directory)
                records[0][field] = value
                self.jsonl(directory / 'target.jsonl',records)
                with self.assertRaises(RuntimeError):
                    analysis.verified_endpoint(directory,'target',summary,2,2)

    def test_scope_partial_fraction_is_not_scope_full_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records,summary = self.endpoint(directory,'scope')
            records[0]['passed_cases'] = 1
            self.jsonl(directory / 'scope.jsonl',records)
            summary['mean_case_fraction'] = .125
            result = analysis.verified_endpoint(directory,'scope',summary,2,2)
            self.assertEqual(result['matrix'][:,:,0].sum(),0)
            self.assertEqual(result['matrix'][:,:,1].mean(),.125)

    def contrast_fixture(self, difference=1):
        reference = {'identity':[('i0',0),('i0',1),('i1',0),('i1',1)],'names':['full_pass'],
                     'matrix':np.zeros((2,2,1))}
        candidate = copy.deepcopy(reference)
        candidate['matrix'] += difference
        return candidate,reference

    def test_two_level_bootstrap_pairs_sign_and_determinism(self):
        candidate,reference = self.contrast_fixture()
        result = analysis.paired_contrast(candidate,reference)
        self.assertEqual(result,[{'metric':'full_pass','difference':1.,'paired_95_interval':[1.,1.]}])
        self.assertEqual(analysis.paired_contrast(reference,candidate)[0]['paired_95_interval'],[-1.,-1.])
        candidate['matrix'][0,0,0] = 0
        self.assertEqual(analysis.paired_contrast(candidate,reference),analysis.paired_contrast(candidate,reference))

    def test_unpaired_primary_or_reordered_rows_cannot_be_treated_as_paired(self):
        candidate,reference = self.contrast_fixture()
        candidate['identity'].reverse()
        with self.assertRaises(RuntimeError):
            analysis.paired_contrast(candidate,reference)

    def test_zero_success_interval_is_empirical_not_a_true_probability_bound(self):
        candidate,reference = self.contrast_fixture(0)
        self.assertEqual(analysis.paired_contrast(candidate,reference)[0]['paired_95_interval'],[0.,0.])

    def test_generation_accounting_includes_endpoints_once_not_cache_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            row = {'completion_tokens':2,'total_cases':3}
            for name in ('initial/target.jsonl','initial/scope.jsonl','branches/0/target.jsonl',
                         'branches/0/scope.jsonl','branches/0/train/update_0001.jsonl',
                         'branches/0/train/update_0001.parts/duplicate.jsonl'):
                self.jsonl(directory / name,[row])
            result = {'training':{'update':1,'train_tokens':1,'nonzero_advantage_tokens':1,'generated_tokens':2}}
            ledger = analysis.generation_cost(directory,result)
            self.assertEqual(ledger['generated_completion_tokens_all_phases'],10)
            self.assertEqual(ledger['non_loss_generation_tokens'],9)
            self.assertEqual(ledger['binary_verifier_test_calls'],15)
            result['training']['generated_tokens'] = 3
            with self.assertRaises(RuntimeError):
                analysis.generation_cost(directory,result)

    def block_fixture(self, exp, complete=True):
        source = Path(__file__).resolve().parents[1]
        block = json.loads((source / analysis.ctl.PLAN).read_text())
        write(exp / analysis.ctl.PLAN,block)
        write(exp / 'raw_results/verge_control_v1/protocol_frozen.json',block)
        results = {}
        if not complete:
            return results
        for r in range(6):
            for label in LABELS:
                directory = exp / 'raw_results' / analysis.ctl.version(label,r)
                initial = {'checkpoint':'synthetic_start'}
                result = {'checkpoint':'synthetic_end'}
                for parent,summary in ((directory / 'initial',initial),(directory / 'branches/0',result)):
                    for kind,n,draws in (('target',64,8),('scope',32,4)):
                        _,summary[kind] = self.endpoint(parent,kind,n=n,draws=draws)
                    write(parent / 'endpoint.json',summary)
                result.update(training={'update':1,'optimizer_steps':1,'train_tokens':524288,
                    'generated_tokens':524288,'nonzero_advantage_tokens':32,'target_successes':0,
                    'first_target_success':None},scope_safe_vs_start=True)
                self.jsonl(directory / 'branches/0/train/update_0001.jsonl',
                           [{'completion_tokens':2048,'total_cases':2}]*256)
                write(directory / 'round_frozen.json',{'initial':initial})
                results[label,r] = result
        return results

    def test_partial_block_cannot_generate_completed_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            self.block_fixture(exp,False)
            with patch.object(analysis.ctl,'EXP',exp), patch.object(analysis.ctl,'ROOT',exp / 'raw_results/verge_control_v1'), patch.object(analysis.ctl,'completed',return_value=None):
                with self.assertRaises(RuntimeError):
                    analysis.collect()

    def test_complete_fixture_keeps_all_rounds_endpoints_and_training_events_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            results = self.block_fixture(exp)
            # Bootstrap tested separately above; avoid 234 duplicate resamplings here.
            def contrasts(candidate,reference,replicates):
                self.assertEqual(replicates,2000)
                self.assertEqual(candidate['identity'],reference['identity'])
                return [{'metric':name,'difference':0.,'paired_95_interval':[0.,0.]} for name in candidate['names']]
            with patch.object(analysis.ctl,'EXP',exp), patch.object(analysis.ctl,'ROOT',exp / 'raw_results/verge_control_v1'), patch.object(analysis.ctl,'completed',side_effect=lambda label,r,deep:results[label,r]), patch.object(analysis,'paired_contrast',side_effect=contrasts):
                result = analysis.collect()
            self.assertEqual(len(result['endpoint_rows']),468)
            self.assertEqual(len(result['within_control_contrasts']),234)
            self.assertEqual(len(result['trajectory_events']),18)
            self.assertEqual(sum(c['loss_tokens'] for c in result['generation_ledger']),9437184)
            self.assertFalse(result['control_block_complete'])
            self.assertFalse(result['suite_complete'])
            self.assertFalse(result['official_test_opened'])
            self.assertFalse(result['cross_primary_paired_comparisons'])
            self.assertIn('not an upper bound',result['statistical_note'])
            self.assertFalse(any(r['observed_rate'] for r in result['endpoint_rows'] if r['metric'] == 'full_pass'))


if __name__ == '__main__':
    unittest.main()
