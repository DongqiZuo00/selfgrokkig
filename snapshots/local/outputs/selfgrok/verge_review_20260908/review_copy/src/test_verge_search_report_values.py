"""Synthetic report-input validation; no Slurm, model or rendering calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_verge_search_analysis import fixture
from verge_search_analysis import analyze
import verge_search_report_values as report
from verge_search_controller import write


def values():
    data=analyze(fixture());data.update(bootstrap_replicates=2000,within_search_contrasts=[],generation_ledger=[])
    events={(e['round'],e['branch_index']):e for e in data['trajectory_events']}
    for r in range(6):
        for b in (1,2,3):
            for kind,names in [('target',report.condition_names()),('scope',['full_pass_rate','mean_case_fraction'])]:
                for metric in names:
                    data['within_search_contrasts'].append({'round':r,'candidate':b,'reference':0,'kind':kind,
                        'metric':metric,'difference':0.,'paired_95_interval':[0.,0.],
                        'identical_policy_by_zero_update_provenance':events[r,b]['optimizer_steps']==0 and events[r,0]['optimizer_steps']==0})
        data['generation_ledger'].append({'round':r,'loss_tokens':2097152,'generated_initial_tokens':1280,
            'generated_training_tokens':2097152,'generated_worker_endpoint_tokens':5120,
            'generated_selected_completion_tokens':128,'generated_tokens_all_phases':2103680,
            'non_loss_generation_tokens':6528,'challenger_generated_tokens':0,
            'rollouts':4288,'binary_verifier_test_calls':8576})
    return data


def allocation_fixture():
    registered={};allocations=[]
    for r in range(6):
        for stage,ids in [('prepare',[(str(100+r*3),None)]),
                          ('worker',[(f'{101+r*3}_{b}',b) for b in range(4)]),
                          ('finish',[(str(102+r*3),None)])]:
            for job,b in ids:
                meta={'version':report.ctl.version(r,b),'round':r,'search_branch_index':b,'stage':stage,'parent_job_id':job.split('_')[0]}
                registered[job]=meta
                allocations.append(dict(meta,job_id=job,state='COMPLETED',allocated_gpus=1,allocated_cpus=8,
                    allocated_cpu_memory_bytes=32*1024**3,elapsed_seconds=3600,gpu_allocation_hours=1.,cpu_allocation_core_hours=8.))
    return {'accounting_complete':True,'missing_individual_accounting_rows':[],
            'allocations':allocations,'gpu_allocation_hours':36.,'cpu_allocation_core_hours':288.},registered


class ReportValueTests(unittest.TestCase):
    def test_complete_matrix_retains_zero_observations_without_claiming_success(self):
        data=values();indexed=report.validate_analysis(data)
        self.assertEqual(len(indexed),456);self.assertEqual(data['right_censored_branches'],24)
        self.assertFalse(data['search_block_complete']);self.assertFalse(data['suite_complete'])

    def test_missing_duplicate_and_changed_denominator_are_rejected(self):
        for change in ('missing','duplicate','denominator','count','tokens'):
            data=values()
            if change=='missing':data['endpoint_rows'].pop()
            elif change=='duplicate':data['endpoint_rows'].append(copy.deepcopy(data['endpoint_rows'][0]))
            else:data['endpoint_rows'][0][{'denominator':'rollouts','count':'observed_count','tokens':'branch_training_loss_tokens'}[change]]=1
            with self.subTest(change=change),self.assertRaises(ValueError):report.validate_analysis(data)

    def test_scope_fraction_cannot_become_a_target_count_or_policy(self):
        for change in ('count','checkpoint','nan','noninteger_full_count'):
            data=values();row=next(p for p in data['endpoint_rows'] if p['kind']=='scope')
            row[{'count':'observed_count','checkpoint':'checkpoint','nan':'observed_rate','noninteger_full_count':'observed_rate'}[change]]={'count':1,'checkpoint':'foreign','nan':float('nan'),'noninteger_full_count':.001}[change]
            with self.subTest(change=change),self.assertRaises(ValueError):report.validate_analysis(data)

    def test_trajectory_event_and_selection_cannot_replace_fixed_endpoint(self):
        for change in ('event','completion','lineage','scope_event'):
            data=values()
            if change=='event':data['trajectory_events'][0]['target_endpoint_full_successes']=1
            elif change=='scope_event':data['trajectory_events'][0]['scope_successes_used_as_target_events']=True
            elif change=='completion':data['round_selections'][0]['completion_used_for_selection']=True
            else:data['round_selections'][0]['selected_checkpoint']='foreign'
            with self.subTest(change=change),self.assertRaises(ValueError):report.validate_analysis(data)

    def test_contrast_and_unique_token_ledger_must_reconcile(self):
        for change in ('reference','difference','interval','identity','cost','missing_contrast'):
            data=values();c=data['within_search_contrasts'][0]
            if change=='cost':data['generation_ledger'][0]['generated_tokens_all_phases']+=1
            elif change=='missing_contrast':data['within_search_contrasts'].pop()
            else:c[{'reference':'reference','difference':'difference','interval':'paired_95_interval','identity':'identical_policy_by_zero_update_provenance'}[change]]={'reference':2,'difference':.2,'interval':[1.,-1.],'identity':not c['identical_policy_by_zero_update_provenance']}[change]
            with self.subTest(change=change),self.assertRaises(ValueError):report.validate_analysis(data)

    def test_cost_includes_shared_gpu_evaluations_and_all_workers(self):
        snapshot,registered=allocation_fixture();result=report.resource_totals(snapshot,registered)
        self.assertEqual(result['stages']['worker']['gpu_allocation_hours'],24.)
        self.assertEqual(result['stages']['prepare']['gpu_allocation_hours'],6.)
        self.assertEqual(result['stages']['finish']['gpu_allocation_hours'],6.)
        self.assertEqual(result['totals']['gpu_allocation_hours'],36.)

    def test_foreign_live_oversized_or_unreconciled_resources_rejected(self):
        for field,value in [('state','RUNNING'),('allocated_gpus',2),('allocated_cpus',9),
                            ('allocated_cpu_memory_bytes',64*1024**3),('gpu_allocation_hours',0.),('round',9)]:
            snapshot,registered=allocation_fixture();snapshot['allocations'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):report.resource_totals(snapshot,registered)
        snapshot,registered=allocation_fixture();snapshot['allocations'].append(snapshot['allocations'][0])
        with self.assertRaises(ValueError):report.resource_totals(snapshot,registered)

    def test_failed_historical_cost_is_never_discarded(self):
        snapshot,registered=allocation_fixture();old=copy.deepcopy(snapshot['allocations'][0])
        old.update(job_id='99',parent_job_id='99',state='FAILED')
        registered['99']={k:old[k] for k in ('version','round','search_branch_index','stage','parent_job_id')}
        snapshot['allocations'].append(old);snapshot['gpu_allocation_hours']+=1;snapshot['cpu_allocation_core_hours']+=8
        self.assertEqual(report.resource_totals(snapshot,registered)['totals']['gpu_allocation_hours'],37.)

    def test_collection_requires_current_finish_success_and_does_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for r in range(6):write(root/'jobs'/f'round_{r:02d}.json',{'prepare':str(100+r*3),
                'workers':str(101+r*3),'finish':str(102+r*3),'branches':[0,1,2,3]})
            snapshot,registered=allocation_fixture()
            with patch.object(report.ctl,'ROOT',root),patch.dict(report.os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_SEARCH_SUITE':'verge_search_v1'}),\
                    patch.object(report.analysis,'collect',return_value=values()),\
                    patch.object(report.resources,'collect',return_value=snapshot),\
                    patch.object(report.resources,'registry',return_value=(registered,[])):
                result=report.collect();self.assertFalse(result['report_rendering_complete']);self.assertFalse(result['suite_complete'])
                snapshot['allocations'][-1]['state']='FAILED'
                with self.assertRaises(RuntimeError):report.collect()


if __name__=='__main__':unittest.main()
