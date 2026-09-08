"""Synthetic six-round endpoint analysis; not empirical model results."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import verge_search_analysis as analysis
import test_verge_control_analysis as native_fixture
from verge_search_controller import write
from verge_search_protocol import PLAN,round_config,worker_config,prepared_round,select_round,version
from verge_search_events import final_event
from verge_round_core import condition_names
from verge_search_analysis import analyze
from test_verge_search_protocol import endpoint

EXP=Path(__file__).resolve().parents[1]


def fixture():
    plan=json.loads((EXP/PLAN).read_text());entries={};previous=None
    for r in range(6):
        cfg=round_config(EXP,r,plan,previous);initial=endpoint(cfg['solver_start']);initial['target']['condition_names']=condition_names()
        frozen=prepared_round(cfg,initial,.5);branches={}
        for b in range(4):
            result=endpoint(f'checkpoints/{version(r,b)}_direct/resume_u0032')
            result['target']['condition_names']=condition_names()
            result['training'].update(generated_tokens=524288,first_target_success=None)
            result.update(search_segment_verified=True,suite_complete=False)
            result['search_event']=final_event(worker_config(cfg,b),result['training'],result['target']);branches[b]=result
        decision=select_round(frozen,branches);completion=copy.deepcopy(initial['target'])
        completion.update(checkpoint=decision['selected']['checkpoint'],rollouts=64,
            keyed_vectors={str(i):[0]*11 for i in range(64)})
        complete={'protocol_version':version(r),'search_round_complete':True,'search_block_complete':False,
            'suite_complete':False,'completed_branches':4,'round_loss_tokens':4*524288,'official_test_opened':False,
            'selected':decision['selected'],'completion_draws':64,'completion_full_successes':0,'completion_used_for_selection':False}
        entries[r]={'frozen':frozen,'branches':branches,'decision':decision,'complete':complete,'completion':completion};previous=complete
    return entries


class AnalysisTests(unittest.TestCase):
    def test_all_six_rounds_preserve_censoring_and_fixed_endpoint_denominators(self):
        result=analyze(fixture());self.assertEqual(len(result['endpoint_rows']),456)
        self.assertEqual(len(result['trajectory_events']),24);self.assertEqual(len(result['round_selections']),6)
        self.assertEqual(result['right_censored_branches'],24);self.assertEqual(result['total_loss_tokens'],12582912)
        self.assertFalse(result['search_block_complete']);self.assertFalse(result['suite_complete'])

    def test_missing_round_or_worker_is_not_zero_success(self):
        entries=fixture();del entries[5]
        with self.assertRaises(ValueError):analyze(entries)
        entries=fixture();del entries[0]['branches'][2]
        with self.assertRaises(ValueError):analyze(entries)

    def test_training_event_does_not_enter_endpoint_rate(self):
        entries=fixture();entry=entries[0];worker=entry['branches'][1]
        worker['training'].update(target_successes=1,optimizer_steps=32,
            first_target_success={'role':'search_target_training','observation_loss_tokens':16384})
        worker['search_event']=final_event(worker_config(entry['frozen']['config'],1),worker['training'],worker['target'])
        entry['decision']=select_round(entry['frozen'],entry['branches']);entry['complete']['selected']=entry['decision']['selected']
        result=analyze(entries);self.assertEqual(result['observed_training_or_endpoint_event_branches'],1)
        self.assertTrue(all(row['observed_count']==0 for row in result['endpoint_rows'] if row['kind']=='target'))

    def test_changed_selection_or_completion_checkpoint_rejected(self):
        for field in ('decision','completion'):
            entries=fixture()
            if field=='decision':entries[0][field]['selected']['name']='changed'
            else:entries[0][field]['checkpoint']='checkpoints/foreign'
            with self.assertRaises(ValueError):analyze(entries)


def disk_fixture(exp, *, all_rounds=True):
    """Native endpoint records and committed-token files, not model sampling."""
    entries=fixture();plan=json.loads((EXP/PLAN).read_text())
    write(exp/PLAN,plan);root=exp/'raw_results/verge_search_v1'
    write(root/'protocol_frozen.json',plan)
    helper=native_fixture.ControlAnalysisTests()
    for r in range(6 if all_rounds else 1):
        entry=entries[r];directory=root/'rounds'/version(r)
        for b,result in entry['branches'].items():
            output=exp/'raw_results'/version(r,b)/'branches/0'
            for kind,n,draws in (('target',64,8),('scope',32,4)):
                _,profile=helper.endpoint(output,kind,n=n,draws=draws)
                if kind=='target':
                    profile.update(counts=[0]*11,rates=[0.]*11,bottleneck_index=0,
                                   keyed_vectors={k:[0]*11 for k in profile['keyed_vectors']})
                else:profile.update(rollouts=128,full_pass_count=0)
                result[kind]=profile
            write(output/'endpoint.json',{k:result[k] for k in ('checkpoint','target','scope')})
            for update in range(1,33):
                helper.jsonl(output/'train'/f'update_{update:04d}.jsonl',
                             [{'completion_tokens':2048,'total_cases':2}]*8)
            # Deliberate cache duplicate: it must not add billed generation.
            helper.jsonl(output/'train/update_0001.parts/cached.jsonl',
                         [{'completion_tokens':2048,'total_cases':2}]*8)
        entry['frozen']['initial']['scope'].update(full_pass_rate=0.,mean_case_fraction=0.,rollouts=128,full_pass_count=0)
        entry['decision']=select_round(entry['frozen'],entry['branches'])
        entry['complete']['selected']=entry['decision']['selected']
        entry['completion']['checkpoint']=entry['decision']['selected']['checkpoint']
        write(directory/'round_frozen.json',entry['frozen'])
        write(directory/'decision.json',entry['decision'])
        write(directory/'completion_profile.json',entry['completion'])
        for kind,count in (('target',512),('scope',128)):
            helper.jsonl(directory/'initial'/(kind+'.jsonl'),
                         [{'completion_tokens':2,'total_cases':2}]*count)
        helper.jsonl(directory/'completion.jsonl',[{'completion_tokens':2,'total_cases':2}]*64)
    return entries


class CollectionTests(unittest.TestCase):
    def test_unique_generation_files_include_shared_evaluation_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);entries=disk_fixture(exp,all_rounds=False)
            directory=exp/'raw_results/verge_search_v1/rounds'/version(0)
            with patch.object(analysis.ctl,'EXP',exp):
                result=analysis.generation_cost(directory,entries[0])
            self.assertEqual(result['loss_tokens'],4*524288)
            self.assertEqual(result['generated_training_tokens'],4*524288)
            self.assertEqual(result['generated_initial_tokens'],1280)
            self.assertEqual(result['generated_worker_endpoint_tokens'],5120)
            self.assertEqual(result['generated_selected_completion_tokens'],128)
            self.assertEqual(result['non_loss_generation_tokens'],6528)
            self.assertEqual(result['rollouts'],4288)
            self.assertEqual(result['binary_verifier_test_calls'],8576)

    def test_missing_committed_group_or_inconsistent_generation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);entries=disk_fixture(exp,all_rounds=False)
            directory=exp/'raw_results/verge_search_v1/rounds'/version(0)
            entries[0]['branches'][0]['training']['generated_tokens']+=1
            with patch.object(analysis.ctl,'EXP',exp),self.assertRaises(ValueError):
                analysis.generation_cost(directory,entries[0])
            entries[0]['branches'][0]['training']['generated_tokens']-=1
            missing=exp/'raw_results'/version(0,0)/'branches/0/train/update_0001.jsonl'
            missing.unlink()
            with patch.object(analysis.ctl,'EXP',exp),self.assertRaises(FileNotFoundError):
                analysis.generation_cost(directory,entries[0])

    def test_collection_checks_all_24_native_endpoints_and_keeps_234_contrasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);entries=disk_fixture(exp)
            def round_done(r,*,deep):
                self.assertTrue(deep);return entries[r]['complete']
            def worker_done(r,b,*,deep):
                self.assertTrue(deep);return entries[r]['branches'][b]
            def contrasts(candidate,reference,replicates):
                self.assertEqual(replicates,2000)
                self.assertEqual(candidate['identity'],reference['identity'])
                return [{'metric':name,'difference':0.,'paired_95_interval':[0.,0.]} for name in candidate['names']]
            with patch.object(analysis.ctl,'EXP',exp),patch.object(analysis.ctl,'ROOT',exp/'raw_results/verge_search_v1'),\
                    patch.object(analysis.ctl,'completed_round',side_effect=round_done) as rounds,\
                    patch.object(analysis.ctl,'completed_worker',side_effect=worker_done) as workers,\
                    patch.object(analysis,'paired_contrast',side_effect=contrasts),\
                    patch.object(analysis,'verified_endpoint',wraps=analysis.verified_endpoint) as endpoints:
                result=analysis.collect()
            self.assertEqual(rounds.call_count,6);self.assertEqual(workers.call_count,24)
            self.assertEqual(endpoints.call_count,48)
            self.assertEqual(len(result['within_search_contrasts']),234)
            self.assertEqual(len(result['endpoint_rows']),456)
            self.assertEqual(sum(c['loss_tokens'] for c in result['generation_ledger']),12582912)
            self.assertFalse(result['search_block_complete']);self.assertFalse(result['suite_complete'])

    def test_collection_stops_before_interpreting_an_incomplete_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);plan=json.loads((EXP/PLAN).read_text())
            write(exp/PLAN,plan);write(exp/'raw_results/verge_search_v1/protocol_frozen.json',plan)
            with patch.object(analysis.ctl,'EXP',exp),patch.object(analysis.ctl,'ROOT',exp/'raw_results/verge_search_v1'),\
                    patch.object(analysis.ctl,'completed_round',return_value=None),\
                    patch.object(analysis.ctl,'completed_worker') as worker:
                with self.assertRaises(RuntimeError):analysis.collect()
                worker.assert_not_called()


if __name__=='__main__':unittest.main()
