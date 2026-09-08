"""Synthetic dispatch and receipt faults; all Slurm submission calls are mocked."""
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_followon_controller as ctl
from test_verge_followon_protocol import bank,plan,EXP


class ControllerTests(unittest.TestCase):
    def fixture(self,exp):
        root=exp/'raw_results/verge_followon_v1';b=bank()
        for cohort in b['cohorts']:cohort['no_full_event_recorded_in_source_cohort']=True
        ctl.write(root/'source_bank.json',b)
        ctl.write(exp/ctl.PLAN,plan())
        ctl.write(exp/'manifests/verge_mistral_round1.json',json.loads((EXP/'manifests/verge_mistral_round1.json').read_text()))
        ctl.write(exp/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json',{'primary_block_complete':True,'outer_rounds':24})
        ctl.write(exp/'raw_results/verge_control_v1/DIRECT_CONTROL_BLOCK_COMPLETE.json',{'control_block_complete':True})
        ctl.write(exp/'raw_results/verge_control_v1/jobs/round_05.json',{'coordinator':'100'})
        for cohort in b['cohorts']:
            for candidate in cohort['candidates']:
                p=exp/candidate['source_checkpoint'];ctl.write(p/'verge_committed.json',{'update':30})
                for name in ('state.pt','adapter_model.safetensors'):(p/name).write_bytes(b'synthetic-only')
        return root

    def test_version_covers_exactly_72(self):
        self.assertEqual(ctl.version(0),'verge_followon_v1_verge_r00_c1')
        self.assertEqual(ctl.version(71),'verge_followon_v1_outcome_r05_c3')
        self.assertEqual(len({ctl.version(i) for i in range(72)}),72)
        for bad in (-1,72,True):
            with self.assertRaises(ValueError):ctl.version(bad)

    def test_predecessor_blocks_submission(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ctl,'EXP',Path(d)), \
                patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_FOLLOWON_SUITE':'verge_followon_v1'}), \
                patch.object(ctl.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'predecessor'):ctl.dispatch()
            run.assert_not_called()

    def test_freeze_and_mapping_before_one_array_and_coordinator(self):
        with tempfile.TemporaryDirectory() as d:
            exp=Path(d);root=self.fixture(exp)
            with patch.object(ctl,'EXP',exp),patch.object(ctl,'ROOT',root), \
                    patch.object(ctl,'dispatch_lock',side_effect=nullcontext), \
                    patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_FOLLOWON_SUITE':'verge_followon_v1'}), \
                    patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(stdout='200\n'),SimpleNamespace(stdout='201\n')]) as run:
                jobs=ctl.dispatch()
                self.assertEqual(jobs['workers'],'200');self.assertEqual(jobs['coordinator'],'201')
                self.assertEqual(len(list((exp/'manifests').glob('verge_followon_v1_*_r*_c*.json'))),72)
                first,second=[call.args[0] for call in run.call_args_list]
                self.assertIn('--array=0-71%2',first);self.assertIn('--dependency=afterok:100',first)
                self.assertIn('--dependency=afterok:200',second)
                self.assertTrue((root/'protocol_frozen.json').is_file())
                self.assertTrue((root/'source_bank_frozen.json').is_file())
                self.assertEqual(ctl.dispatch(),jobs);self.assertEqual(run.call_count,2)
                frozen=ctl.read(root/'protocol_frozen.json');frozen['tie_rule']='changed'
                ctl.write(root/'protocol_frozen.json',frozen)
                with self.assertRaisesRegex(RuntimeError,'Frozen record changed'):ctl.dispatch()
                self.assertEqual(run.call_count,2)

    def test_uncertain_submission_never_blindly_retries(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ctl,'ROOT',Path(d)), \
                patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(stdout='uncertain')) as run:
            with self.assertRaisesRegex(RuntimeError,'Uncertain'):ctl.submit('workers','100')
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):ctl.submit('workers','100')
            self.assertEqual(run.call_count,1)

    def test_confirmed_intent_recovers_lost_outer_receipt(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ctl,'ROOT',Path(d)), \
                patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(stdout='200;cluster\n')) as run:
            self.assertEqual(ctl.submit('workers','100'),'200')
            self.assertEqual(ctl.submit('workers','100'),'200');self.assertEqual(run.call_count,1)
            with self.assertRaisesRegex(RuntimeError,'arguments changed'):ctl.submit('workers','101')

    def test_pre_success_excludes_prior_same_arm_events_without_dropping_candidates(self):
        b=bank()
        for c in b['cohorts']:c['no_full_event_recorded_in_source_cohort']=True
        b['cohorts'][0]['no_full_event_recorded_in_source_cohort']=False
        result=ctl.analysis_strata(b);rows=result['cohorts']
        self.assertFalse(rows[0]['pre_success_analysis']);self.assertFalse(rows[4]['pre_success_analysis'])
        self.assertTrue(rows[1]['pre_success_analysis']);self.assertTrue(rows[5]['pre_success_analysis'])
        self.assertEqual(sum(r['candidates_retained'] for r in rows),72)
        self.assertFalse(result['candidate_filter_applied'])

    def test_coordinator_does_not_mark_analysis_or_book_complete(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ctl,'ROOT',Path(d)), \
                patch.object(ctl,'completed',return_value={'synthetic':True}) as completed:
            result=ctl.coordinate();self.assertEqual(completed.call_count,72)
            self.assertFalse(result['followon_block_complete']);self.assertFalse(result['suite_complete'])
            self.assertEqual(result['candidate_loss_tokens'],18874368)

    def test_coordinator_refuses_missing_candidate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(ctl,'ROOT',Path(d)), \
                patch.object(ctl,'completed',side_effect=lambda i,**kw:None if i==7 else {'synthetic':True}):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):ctl.coordinate()
            self.assertFalse((Path(d)/'FOLLOWONS_FINISHED.json').exists())

    def test_worker_script_resource_and_index_bounds(self):
        worker=(EXP/'scripts/verge_followon_workers.sbatch').read_text()
        coordinator=(EXP/'scripts/verge_followon_coordinate.sbatch').read_text()
        for text in ('--gpus=b200:1','--cpus-per-task=8','--mem=32gb','index >= 72','ROLLOUT_MAX_COMPLETION_TOKENS=2048'):
            self.assertIn(text,worker)
        self.assertIn('--mem=4gb',coordinator);self.assertNotIn('--gpus',coordinator)


if __name__=='__main__':unittest.main()
