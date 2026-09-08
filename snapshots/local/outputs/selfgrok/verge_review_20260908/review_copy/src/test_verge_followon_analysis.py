"""Synthetic all-candidate analysis; no model, Slurm or new evaluation draws."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_followon_analysis as analysis
from verge_followon_controller import analysis_strata,version
from verge_followon_events import final_event,BUDGET
from verge_followon_protocol import candidate_from_bank
from test_verge_followon_protocol import bank


def profile(n,full=0):
    counts=[full]*11
    return {'rollouts':n,'counts':counts,'rates':[x/n for x in counts],
            'condition_names':analysis.conditions()}


def fixture():
    b=bank();records={}
    for cohort in b['cohorts']:cohort['no_full_event_recorded_in_source_cohort']=True
    for i in range(72):
        c=candidate_from_bank(b,i)
        state={'update':16,'train_tokens':BUDGET,'target_successes':0,'first_target_success':None,
               'optimizer_steps':0,'nonzero_advantage_tokens':0,'generated_tokens':BUDGET,'rollouts':128}
        initial={'checkpoint':c['source_checkpoint'],'target':profile(512),
                 'scope':{'full_pass_rate':0.,'mean_case_fraction':0.},'training':{'target_successes':0}}
        result={'source_candidate':copy.deepcopy(c),'training':state,'target':profile(512),'completion':profile(64),
            'scope':{'full_pass_rate':0.,'mean_case_fraction':0.},'scope_safe_vs_start':True,
            'checkpoint':f'checkpoints/{version(i)}_direct/resume_u0016','followon_segment_verified':True,
            'followon_only':True,'suite_complete':False}
        result['followon_event']=final_event(c,state,result['target'],result['completion'])
        records[i]={'initial':initial,'result':result}
    return b,analysis_strata(b),records


class AnalysisTests(unittest.TestCase):
    def aggregate(self,data,**filters):
        filters=dict(stratum='pre_success',source_arm='all_arms',strategy='condition',top_k=1,**filters)
        return next(r for r in data['descriptive_aggregates'] if all(r[k]==v for k,v in filters.items()))

    def test_all_censored_coverage_and_null_concordance(self):
        data=analysis.analyze(*fixture())
        self.assertEqual(len(data['endpoint_rows']),72*37)
        self.assertEqual(len(data['trajectory_events']),72);self.assertEqual(len(data['cohorts']),24)
        self.assertEqual(len(data['descriptive_aggregates']),135)
        agg=self.aggregate(data);self.assertEqual(agg['right_censored'],72)
        self.assertEqual(agg['mean_cohort_top_k_event_fraction'],0.)
        self.assertIsNone(agg['pair_weighted_descriptive_concordance'])
        post=next(r for r in data['descriptive_aggregates'] if r['stratum']=='post_ignition')
        self.assertEqual(post['cohorts'],0);self.assertIsNone(post['mean_cohort_top_k_event_fraction'])
        self.assertFalse(data['suite_complete']);self.assertFalse(data['followon_block_complete'])

    def test_training_event_is_not_final_endpoint_success(self):
        b,s,r=fixture();v=r[0]['result'];v['training'].update(target_successes=1,optimizer_steps=16,
            first_target_success={'role':'followon_target_training','observation_loss_tokens':16384})
        v['followon_event']=final_event(v['source_candidate'],v['training'],v['target'],v['completion'])
        data=analysis.analyze(b,s,r);c=data['cohorts'][0]
        self.assertEqual(c['observed_events_within_budget'],1)
        self.assertEqual(c['strategies']['condition']['observed_order_concordance']['concordance'],0.)
        self.assertEqual(c['strategies']['random']['top_k'][0]['mean_observed_event_fraction'],1/3)
        self.assertEqual(self.aggregate(data)['within_cohort_comparable_pairs'],2)
        self.assertTrue(all(p['observed_count']==0 for p in data['endpoint_rows'] if p['metric']=='full_pass'))

    def test_final_event_at_budget_kept_but_event_censor_ties_unordered(self):
        b,s,r=fixture();v=r[2]['result'];v['target']=profile(512,1)
        v['followon_event']=final_event(v['source_candidate'],v['training'],v['target'],v['completion'])
        data=analysis.analyze(b,s,r);c=data['cohorts'][0]
        self.assertEqual(c['strategies']['condition']['top_k'][0]['mean_any_observed_event_indicator'],1.)
        self.assertIsNone(c['strategies']['condition']['observed_order_concordance']['concordance'])
        self.assertEqual(data['trajectory_events'][2]['first_event']['role'],'followon_target_selection')

    def test_prior_source_event_keeps_all_candidates_and_is_not_new_event(self):
        b,s,r=fixture();b['cohorts'][0]['no_full_event_recorded_in_source_cohort']=False
        c=b['cohorts'][0]['candidates'][0];c['initial_target_full_successes']=1
        r[0]['initial']['target']=profile(512,1);v=r[0]['result'];v['source_candidate']=copy.deepcopy(c)
        v['followon_event']=final_event(c,v['training'],v['target'],v['completion'])
        data=analysis.analyze(b,analysis_strata(b),r)
        self.assertEqual(len(data['trajectory_events']),72)
        self.assertEqual(self.aggregate(data)['candidates'],54)
        self.assertTrue(data['trajectory_events'][0]['prior_source_event_recorded'])
        self.assertFalse(data['trajectory_events'][0]['event_observed'])

    def test_missing_run_is_not_a_negative(self):
        b,s,r=fixture();del r[71]
        with self.assertRaisesRegex(ValueError,'All 72'):analysis.analyze(b,s,r)

    def test_posthoc_stratum_and_foreign_source_rejected(self):
        b,s,r=fixture();s['cohorts'][0]['pre_success_analysis']=False
        with self.assertRaisesRegex(ValueError,'strata'):analysis.analyze(b,s,r)
        s=analysis_strata(b);r[0]['result']['source_candidate']['candidate_index']=2
        with self.assertRaisesRegex(ValueError,'identity'):analysis.analyze(b,s,r)

    def test_profile_or_event_change_rejected(self):
        b,s,r=fixture();r[0]['result']['target']['rates'][0]=.5
        with self.assertRaisesRegex(ValueError,'profile'):analysis.analyze(b,s,r)
        b,s,r=fixture();r[0]['result']['followon_event']['event_observed']=True
        with self.assertRaisesRegex(ValueError,'event/censoring'):analysis.analyze(b,s,r)

    def test_collect_waits_without_sampling_or_zero_filling(self):
        with tempfile.TemporaryDirectory() as d,patch.object(analysis.ctl,'ROOT',Path(d)), \
                patch.object(analysis.ctl,'completed') as completed:
            with self.assertRaisesRegex(RuntimeError,'Wait'):analysis.collect()
            completed.assert_not_called()


class NativeAndCostTests(unittest.TestCase):
    def native_fixture(self,directory):
        from verge_independent_sampling import PROTOCOL
        result={}
        for kind,n,draws in (('target',64,8),('scope',32,4),('completion',64,1)):
            request={'prompt':[[1]]*n,'n':draws,'seed':1000}
            records=[{'instance_id':f'item{i//draws}','instance_index':i//draws,'rollout_index':i%draws,
                'sampling_protocol':PROTOCOL,'sampling_parent_seed':1000+i//draws*draws,'sampling_child_seed':1000+i,
                'prompt_token_ids':[1],'completion_token_ids':[2,3],'completion_tokens':2,'max_tokens':2048,
                'reward':int(i==0),'passed_cases':int(i==0),'total_cases':1} for i in range(n*draws)]
            response={'backend_protocol':PROTOCOL,'prompt_parent_seeds':[1000+i*draws for i in range(n)],
                'child_seed_count':n*draws,'choices':[{'index':i} for i in range(n*draws)]}
            (directory/f'{kind}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (directory/f'{kind}.response.json').write_text(json.dumps({'request':request,'response':response,'backend_protocol':PROTOCOL}))
            if kind=='scope':result[kind]={'full_pass_rate':1/(n*draws),'mean_case_fraction':1/(n*draws)}
            else:
                result[kind]=profile(n*draws,1)
                result[kind]['keyed_vectors']={f"{r['instance_id']}::{r['rollout_index']}":[r['reward']]*11 for r in records}
        (directory/'completion_profile.json').write_text(json.dumps(result['completion']))
        return result

    def test_three_native_profiles_and_non_nested_completion_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);result=self.native_fixture(directory)
            analysis.verify_endpoints(directory,result)
            result['completion']['keyed_vectors']['item0::0'][0]=0
            with self.assertRaisesRegex(ValueError,'nested profile'):analysis.verify_endpoints(directory,result)

    def test_generation_cost_excludes_reused_source_and_counts_completion_once(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d);result=self.native_fixture(directory);(directory/'train').mkdir()
            training=[{'completion_tokens':2048,'total_cases':1} for _ in range(128)]
            (directory/'train/update_0001.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in training))
            result['training']={'update':1,'train_tokens':BUDGET,'generated_tokens':BUDGET,
                                'nonzero_advantage_tokens':0,'rollouts':128}
            ledger=analysis.generation_cost(directory,result)
            self.assertEqual(ledger['generated_completion_tokens_all_phases'],BUDGET+2*(512+128+64))
            self.assertEqual(ledger['non_loss_generation_tokens'],1408)
            self.assertEqual(ledger['generated_initial_endpoint_tokens'],0)
            self.assertEqual(ledger['raw_rollouts'],832)
            result['training']['rollouts']=127
            with self.assertRaisesRegex(ValueError,'generation ledger'):analysis.generation_cost(directory,result)


if __name__=='__main__':unittest.main()
