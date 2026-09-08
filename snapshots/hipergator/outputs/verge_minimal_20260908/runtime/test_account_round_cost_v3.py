"""Accounting-tool fixtures only; not Solver/Challenger training-path tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import account_round_cost_v2 as v2
from account_round_cost_v3 import audit, verify_baseline_reuse


class ReusedBaselineAccounting(unittest.TestCase):
    def setUp(self):
        parent = v2.ROOT / 'evidence/cost_cpu_fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='reuse_fixture_', dir=parent)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.old, self.new = self.root / 'runs/source', self.root / 'runs/reused'
        records, raw_sources = [], []
        for chunk in range(32):
            ids = [[10 + chunk, 100 + slot, 2] for slot in range(8)]
            strings = [f'SYNTHETIC_ACCOUNTING_FIXTURE_{chunk}_{slot}' for slot in range(8)]
            path = self.old / f'base/raw_evaluation/chunk_{chunk}.json'
            self.write(path, {'mode': 'SYNTHETIC_ACCOUNTING_FIXTURE', 'completion_token_ids': ids,
                              'verifier_completions': strings, 'generated_tokens': 24, 'n': 8})
            sha = v2.digest(path.read_bytes())
            raw_sources.append({'path': str(path), 'file_sha256': sha})
            records.extend({'instance_id': f'instance_{chunk // 4}', 'sample_id': f'sample_{(chunk % 4) * 8 + slot}',
                            'generation_evidence_id': str(path) + f'#sample{slot}', 'raw_generation_sha256': sha,
                            'completion_token_ids': ids[slot], 'raw_completion': strings[slot]} for slot in range(8))
        self.original = {'baseline_contract_sha256': 'SYNTHETIC_CONTRACT', 'checkpoints': {'base': 'SYNTHETIC_CHECKPOINT'},
                         'evaluations': {'base': {'records': records}},
                         'execution': {'no_training': True, 'actual_generation_tokens': 768}}
        self.write(self.old / 'base/fragment.json', self.original)
        self.write(self.old / 'plan.json', {'mode': 'SYNTHETIC_ACCOUNTING_FIXTURE'})
        self.write(self.old / 'READY.json', {'mode': 'SYNTHETIC_ACCOUNTING_FIXTURE'})
        self.report = {'source_preparation': str(self.old), 'baseline_contract_sha256': 'SYNTHETIC_CONTRACT',
                       'evaluation_cache_hit': True, 'records_copied_without_any_change': True,
                       'reused_rollout_slots': 256, 'new_evaluation_rollouts': 0, 'new_evaluation_generated_tokens': 0,
                       'original_evaluation_generated_tokens': 768, 'raw_sources': raw_sources}
        for key, name in (('source_fragment', 'base/fragment.json'), ('source_plan', 'plan.json'), ('source_ready', 'READY.json')):
            path = self.old / name
            self.report[key] = {'path': str(path), 'file_sha256': v2.digest(path.read_bytes())}
        self.copied = copy.deepcopy(self.original)
        self.copied['original_execution'] = copy.deepcopy(self.original['execution'])
        self.copied['execution'].update(actual_generation_tokens=0, new_evaluation_rollouts=0, reused_rollout_slots=256)
        self.flush()

    def tearDown(self):
        assert self.root.is_relative_to((v2.ROOT / 'evidence/cost_cpu_fixtures').resolve())
        self.temporary.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def flush(self):
        self.copied['baseline_reuse'] = copy.deepcopy(self.report)
        self.write(self.new / 'baseline_reuse.json', self.report)
        self.write(self.new / 'base/fragment.json', self.copied)

    def verify(self):
        return verify_baseline_reuse(self.root, 'reused', source_preparation='source')

    def test_exact_original_32_chunks_256_slots_charge_zero_increment(self):
        result = self.verify()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['original_generated_tokens'], 768)
        self.assertEqual(result['new_baseline_generated_tokens'], 0)
        self.assertEqual(result['reused_raw_sources'], 32)

    def test_changed_original_raw_bytes_are_detected(self):
        path = Path(self.report['raw_sources'][0]['path'])
        raw = v2.read(path)
        raw['completion_token_ids'][0][0] = 999
        self.write(path, raw)
        with self.assertRaisesRegex(ValueError, 'SHA changed'):
            self.verify()

    def test_self_consistent_copied_record_cannot_disagree_with_actual_raw_ids(self):
        self.original['evaluations']['base']['records'][0]['completion_token_ids'] = [999, 100, 2]
        self.copied['evaluations'] = copy.deepcopy(self.original['evaluations'])
        self.write(self.old / 'base/fragment.json', self.original)
        self.report['source_fragment']['file_sha256'] = v2.digest((self.old / 'base/fragment.json').read_bytes())
        self.flush()
        with self.assertRaisesRegex(ValueError, 'actual original token IDs'):
            self.verify()

    def test_zero_claim_does_not_hide_new_baseline_raw_files(self):
        self.write(self.new / 'base/raw_evaluation/unexpected.json', {'completion_token_ids': [[10, 2]], 'generated_tokens': 2})
        with self.assertRaisesRegex(ValueError, 'new baseline raw files exist'):
            self.verify()

    def test_duplicate_origin_list_does_not_establish_complete_reuse(self):
        self.report['raw_sources'][-1] = self.report['raw_sources'][0]
        self.flush()
        with self.assertRaisesRegex(ValueError, 'duplicate source'):
            self.verify()

    def test_unknown_array_identity_or_missing_tasks_cannot_finalize(self):
        result = audit(self.root, '', None)
        self.assertFalse(result['round3_array_identity_known'])
        self.assertFalse(result['scheduler']['all_named_jobs_terminal'])
        self.assertNotEqual(result['status'], 'complete')
        self.assertEqual(result['validation_scope']['training_path_test_count_claimed_by_this_report'], None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
