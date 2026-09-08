"""Independent read-only reconciliation of all-candidate exported CSV values."""
import argparse
import csv
from pathlib import Path
from verge_followon_controller import read
from verge_control_artifacts import same_cell
from verge_followon_report import validate_analysis

COLUMNS={
    'condition_profiles':['array_index','source_arm','source_round','candidate_index','endpoint','metric','observed_rate','rollouts',
        'kind','observed_count','pre_success_analysis','followon_loss_tokens','source_endpoint_reused',
        'version','source_cohort','source_solver_initial','checkpoint','source_artifact'],
    'trajectory_events':['array_index','source_arm','source_round','candidate_index','event_observed','observation_loss_tokens',
        'right_censored','prior_source_event_recorded','pre_success_analysis','planned_loss_tokens','executed_loss_tokens','budget_complete',
        'target_training_full_successes','target_endpoint_full_successes','completion_full_successes','first_event_role',
        'first_event_update','first_event_group','generated_tokens_through_first_group','condition_rung_zero_based','condition_gain',
        'raw_endpoint_condition_gain','uncertainty_score','selected_in_primary','scope_safe_in_primary','eligible_in_primary',
        'scope_safe_vs_source_observed','optimizer_steps','version','source_cohort','source_solver_initial','source_checkpoint','final_checkpoint'],
    'cohort_rankings':['source_arm','source_round','strategy','top_k','mean_observed_event_fraction','mean_any_observed_event_indicator',
        'concordance','within_cohort_comparable_pairs','concordant_pair_credit','allowed_order_count','pre_success_analysis',
        'source_cohort','source_solver_initial'],
    'cohort_survival':['source_arm','source_round','loss_tokens','at_risk','observed_events','right_censored','survival_estimate',
        'pre_success_analysis','source_cohort','source_solver_initial'],
    'descriptive_aggregates':['stratum','source_arm','strategy','top_k','cohorts','candidates','observed_events','right_censored',
        'sum_cohort_top_k_event_fractions','mean_cohort_top_k_event_fraction','sum_cohort_top_k_any_event_indicators',
        'mean_cohort_top_k_any_event_indicator','within_cohort_comparable_pairs','concordant_pair_credit','pair_weighted_descriptive_concordance'],
    'generation_costs':['array_index','loss_tokens','generated_training_tokens','generated_target_endpoint_tokens','generated_scope_endpoint_tokens',
        'generated_completion_tokens','generated_completion_tokens_all_phases','non_loss_generation_tokens','nonzero_advantage_tokens',
        'generated_initial_endpoint_tokens','raw_rollouts','binary_verifier_test_calls','source_endpoint_reused_without_generation',
        'uncommitted_failed_attempt_generation_included','version'],
    'slurm_allocations':['job_id','candidate_array_index','stage','state','elapsed_seconds','allocated_cpus','allocated_gpus',
        'allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours','batch_max_rss_bytes','parent_job_id','version'],
}


def expected_tables(bundle):
    d=bundle['analysis'];validate_analysis(d)
    profiles=[];events=[];ranks=[];survival=[]
    for r in d['endpoint_rows']:
        stem=f"raw_results/{r['source_cohort']}/branches/{r['candidate_index']}" if r['endpoint']=='source' else f"raw_results/{r['version']}/branches/0"
        profiles.append(dict(r,source_artifact=f"{stem}/{r['kind']}.jsonl"))
    for r in d['trajectory_events']:
        e=r['first_event'] or {}
        events.append(dict(r,first_event_role=e.get('role'),first_event_update=e.get('update'),
                          first_event_group=e.get('group_index'),generated_tokens_through_first_group=e.get('generated_tokens_through_group')))
    for c in d['cohorts']:
        context={k:c[k] for k in ('source_arm','source_round','source_cohort','source_solver_initial','pre_success_analysis')}
        pairs=c['strategies']['condition']['observed_order_concordance']['comparable_pairs']
        for label,s in c['strategies'].items():
            for top in s['top_k']:
                concordance=s['expected_concordance_over_uniform_orders'] if label=='random' else s['observed_order_concordance']['concordance']
                credit=pairs*.5 if label=='random' else s['observed_order_concordance']['concordant_pair_credit']
                ranks.append(dict(context,**top,strategy=label,allowed_order_count=s['allowed_order_count'],
                                  within_cohort_comparable_pairs=pairs,concordance=concordance,concordant_pair_credit=credit))
        survival.extend(dict(context,**p) for p in c['within_cohort_descriptive_survival'])
    return {'condition_profiles':profiles,'trajectory_events':events,'cohort_rankings':ranks,
            'cohort_survival':survival,'descriptive_aggregates':d['descriptive_aggregates'],
            'generation_costs':d['generation_ledger'],
            'slurm_allocations':[dict(r,version=r.get('version')) for r in bundle['resources']['allocations']]}


def validate_csv(bundle,directory):
    expected=expected_tables(bundle);receipt=read(directory/'csv_export_receipt.json')
    if (receipt.get('protocol')!='verge_followon_v1_csv_export' or receipt.get('synthetic') is not bundle.get('synthetic')
            or receipt.get('suite_complete') is not False or receipt.get('followon_block_complete') is not False):
        raise ValueError('Wrong follow-on CSV export receipt')
    entries={r['name']:r for r in receipt['tables']}
    if set(entries)!=set(COLUMNS) or len(receipt['tables'])!=7:raise ValueError('Missing or duplicate CSV table receipt')
    for name,columns in COLUMNS.items():
        rows=expected[name];e=entries[name]
        if (e['rows']!=len(rows) or e['columns']!=len(columns) or e.get('typed_values_unchanged') is not True
                or e.get('preview_rendered') is not True or e.get('preview_columns')!=8):
            raise ValueError('Exported table dimensions or preview contract differ')
        with (directory/f'{name}.csv').open(newline='',encoding='utf-8') as handle:saved=list(csv.reader(handle))
        if len(saved)!=len(rows)+1 or saved[0]!=columns:raise ValueError('Incomplete CSV shape or columns')
        for actual,row in zip(saved[1:],rows):
            if len(actual)!=len(columns) or any(not same_cell(v,row[k]) for v,k in zip(actual,columns)):
                raise ValueError(f'CSV value differs from its source: {name}')
    return {'csv_tables_verified':7,'source_values_unchanged':True,'synthetic':bundle['synthetic'],
            'followon_block_complete':False,'suite_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle');p.add_argument('directory');a=p.parse_args()
    print(validate_csv(read(Path(a.bundle)),Path(a.directory)))
