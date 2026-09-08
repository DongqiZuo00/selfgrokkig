"""Synthetic CPU controls for cost arithmetic/provenance; never experiment data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from account_round_cost_v2 import ROOT, audit, native_counts, parse_scheduler, raw_candidates


def scheduler_line(job, state='COMPLETED', seconds=60, start='2026-09-08T10:00:00',
                   end='2026-09-08T10:01:00', tres='gres/gpu=1,gres/gpu:b200=1,mem=64G'):
    return '|'.join((job, '9000', state, str(seconds), tres, '0:0', start, end, start, 'hpg-b200'))


class Arithmetic(unittest.TestCase):
    def test_native_solver_and_challenger_shapes_include_eos(self):
        self.assertEqual(native_counts({'completion_token_ids': [[10, 2], [11, 12]], 'n': 2,
                                        'max_new_tokens': 2, 'generated_tokens': 4})['native_generated_tokens'], 4)
        self.assertEqual(native_counts({'completion_token_ids': [10, 11, 2], 'generated_tokens': 3})['native_generated_tokens'], 3)

    def test_post_eos_padding_is_excluded_and_record_mismatch_exposed(self):
        value = native_counts({'completion_token_ids': [[10, 2, 2, 2]], 'generated_tokens': 4})
        self.assertEqual(value['native_generated_tokens'], 2)
        self.assertEqual(value['raw_saved_id_count'], 4)
        self.assertEqual(len(value['issues']), 2)

    def test_invalid_native_ids_and_cap_count_errors(self):
        for ids in ([], [[True]], [[-1]], [[10], 11]):
            with self.assertRaises(ValueError):
                native_counts({'completion_token_ids': ids})
        value = native_counts({'completion_token_ids': [10, 2], 'generated_tokens': 2, 'max_new_tokens': 1, 'n': 8})
        self.assertEqual(len(value['issues']), 2)

    def test_scheduler_missing_pending_and_empty_never_mean_complete(self):
        self.assertFalse(parse_scheduler('', {'100'})['all_named_jobs_terminal'])
        data = scheduler_line('100', state='PENDING', seconds=0, start='Unknown', end='Unknown', tres='')
        result = parse_scheduler(data, {'100', '101'})
        self.assertFalse(result['all_named_jobs_terminal'])
        self.assertEqual(result['missing_expected_jobs'], ['101'])
        self.assertEqual(result['allocated_b200_hours'], 0)

    def test_array_steps_and_generic_typed_gpu_counts_not_double_counted(self):
        text = '\n'.join((scheduler_line('100_0'), scheduler_line('100_1'),
                          scheduler_line('100_0.batch'), scheduler_line('100_[2-3]')))
        result = parse_scheduler(text, {'100_0', '100_1'})
        self.assertEqual(result['allocated_b200_hours'], 120 / 3600)
        self.assertEqual(result['peak_named_job_allocated_gpus'], 2)
        self.assertEqual(result['peak_named_job_allocated_memory_gib'], 128)
        self.assertEqual(len(result['ignored_rows']), 2)

    def test_adjacent_allocations_do_not_overlap_and_failed_time_counts(self):
        text = '\n'.join((scheduler_line('100', state='FAILED'),
                          scheduler_line('101', start='2026-09-08T10:01:00', end='2026-09-08T10:02:00')))
        result = parse_scheduler(text, {'100', '101'})
        self.assertTrue(result['all_named_jobs_terminal'])
        self.assertEqual(result['peak_named_job_allocated_gpus'], 1)
        self.assertEqual(result['allocated_b200_hours'], 120 / 3600)

    def test_duplicate_scheduler_rows_require_reconciliation(self):
        result = parse_scheduler(scheduler_line('100') + '\n' + scheduler_line('100'), {'100'})
        self.assertFalse(result['all_named_jobs_terminal'])
        self.assertTrue(result['issues'])


class SourceAccounting(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'evidence/cost_cpu_fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='explicit_fixture_', dir=parent)
        self.root = Path(self.temporary.name).resolve()
        assert self.root.is_relative_to(ROOT.resolve())

    def tearDown(self):
        # Verify the exact recursive cleanup target before TemporaryDirectory removes it.
        assert self.root.is_relative_to((ROOT / 'evidence/cost_cpu_fixtures').resolve())
        self.temporary.cleanup()

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding='utf-8')
        return target

    def raw(self, path):
        return self.write(path, {'mode': 'SYNTHETIC_UNIT_FIXTURE', 'completion_token_ids': [[10, 2]],
                                'generated_tokens': 2, 'n': 1, 'max_new_tokens': 2})

    def test_identical_independent_sources_count_twice_fragment_copy_adds_zero(self):
        first = self.raw('runs/stage/samples/a.json')
        self.raw('runs/stage/samples/b.json')
        self.write('runs/stage/fragment.json', {'evaluations': {'copied': {'records': [
            {'generation_evidence_id': str(first) + '#sample0', 'raw_generation_sha256': hashlib.sha256(first.read_bytes()).hexdigest(),
             'completion_token_ids': [10, 2]}]}}})
        result = audit(self.root, scheduler_line('100'), [dict(job='100', label='fixture', directory='stage', kind='stage')])
        self.assertEqual(result['actual_saved_native_generated_tokens'], 4)
        self.assertEqual(result['unique_generation_source_files'], 2)
        self.assertFalse(result['issues'])

    def test_atomic_rewrite_is_one_source_and_orphan_valid_temporary_remains_visible(self):
        first = self.raw('runs/stage/samples/a.json')
        self.raw('runs/stage/samples/a.json.writing')
        self.raw('runs/stage/samples/b.json.writing')
        candidates = raw_candidates(first.parent, '*.json')
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[first.resolve()], first)
        result = audit(self.root, scheduler_line('100', state='RUNNING'), [dict(job='100', label='fixture', directory='stage', kind='stage')])
        self.assertEqual(result['actual_saved_native_generated_tokens'], 4)
        self.assertEqual(sum(s['from_atomic_temporary'] for s in result['source_files']), 1)

    def test_proof_reuses_counted_sample_and_does_not_add_generation(self):
        first = self.raw('runs/stage/samples/a.json')
        self.write('runs/stage/one_binary_update_proof.json', {'selected_stage': {'raw_path': str(first)},
                     'update': {'tokens_used': 2, 'optimizer_step': True}})
        result = audit(self.root, scheduler_line('100'), [dict(job='100', label='fixture', directory='stage', kind='stage')])
        self.assertEqual(result['actual_saved_native_generated_tokens'], 2)
        self.assertEqual(result['proof_updates'][0]['additional_generated_tokens'], 0)
        self.assertTrue(result['proof_updates'][0]['source_count_verified'])

    def test_bad_reported_count_keeps_actual_count_and_exposes_problem(self):
        self.write('runs/stage/samples/a.json', {'completion_token_ids': [[10, 2]], 'generated_tokens': 999})
        result = audit(self.root, scheduler_line('100'), [dict(job='100', label='fixture', directory='stage', kind='stage')])
        self.assertEqual(result['actual_saved_native_generated_tokens'], 2)
        self.assertTrue(result['issues'])
        self.assertNotEqual(result['status'], 'complete')

    def test_completed_branch_separates_training_budget_probe_cost_and_drain(self):
        rows = [self.raw(f'runs/branch/execution/raw_training/{index}.json') for index in range(2)]
        self.raw('runs/branch/execution/raw_evaluation/eval.json')
        events, previous = [], '0' * 64
        for index, path in enumerate(rows):
            for kind, payload in [('generation_attempt', {'drain': True}), ('generation_source',
                    {'raw_path': str(path), 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'target': bool(index)})]:
                item = {'run_id': 'SYNTHETIC_UNIT_FIXTURE', 'sequence': len(events), 'previous_sha256': previous,
                        'kind': kind, 'payload': payload}
                previous = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
                events.append({**item, 'sha256': previous})
        journal = self.root / 'runs/branch/execution/training_journal.jsonl'
        journal.write_text('\n'.join(json.dumps(event) for event in events) + '\n')
        fragment = {'execution': {'training_generated_tokens': 4, 'backend_tokens_all_generation_calls': 6, 'actual_optimizer_steps': 0},
                    'branches': {'g1': {'phases': [{'generated_tokens': 4, 'nonzero_advantage_tokens': 0}]}}, 'evaluations': {}}
        self.write('runs/branch/fragment.json', fragment)
        self.write('runs/branch/COMPLETE.json', {'status': 'complete'})
        spec = dict(job='100', label='fixture', directory='branch', kind='branch', branch='g1', expected_budget=4)
        result = audit(self.root, scheduler_line('100'), [spec])
        self.assertFalse(result['issues'])
        self.assertEqual(result['generation_totals']['solver_training'], 4)
        self.assertEqual(result['generation_totals']['solver_evaluation'], 2)
        self.assertEqual(result['runs'][0]['training_by_task_kind'], {'target': 2, 'stage': 2, 'unknown': 0})
        self.assertEqual(result['runs'][0]['boundary_drain_tokens'], 4)
        self.assertTrue(result['runs'][0]['budget_check_passed'])
        fragment['execution']['training_generated_tokens'] = 999
        self.write('runs/branch/fragment.json', fragment)
        self.assertTrue(audit(self.root, scheduler_line('100'), [spec])['issues'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
