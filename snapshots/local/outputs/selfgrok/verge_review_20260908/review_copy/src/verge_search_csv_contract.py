"""Independently read every exported search CSV cell, without draws or hashes."""
import argparse
import csv
import json
from pathlib import Path
from verge_control_artifacts import same_cell
from verge_search_report_values import validate_analysis

COLUMNS={
 'condition_profiles':['round','endpoint','branch_index','kind','metric','observed_count','rollouts','observed_rate',
   'branch_training_loss_tokens','checkpoint','round_version','source_artifact'],
 'paired_contrasts':['round','candidate','reference','kind','metric','difference','paired_95_lower','paired_95_upper',
   'identical_policy_by_zero_update_provenance','bootstrap_replicates'],
 'trajectory_events':['round','branch_index','event_observed','observation_loss_tokens','right_censored',
   'target_training_full_successes','target_endpoint_full_successes','optimizer_steps','updates',
   'planned_loss_tokens','executed_loss_tokens','budget_complete','first_event_role','first_event_update','first_event_group',
   'generated_tokens_through_first_group','training_rollouts_used_in_endpoint_rate','scope_successes_used_as_target_events',
   'clock_is_total_search_compute','version','checkpoint'],
 'round_selections':['round','selected_name','selection_condition','selected_gain','completion_full_successes','completion_draws',
   'completion_used_for_selection','challenger_updated','round_loss_tokens_all_four_branches',
   'eligible_branches_json','selection_probabilities_json','selected_checkpoint'],
 'generation_costs':['round','loss_tokens','generated_training_tokens','generated_initial_tokens','generated_worker_endpoint_tokens',
   'generated_selected_completion_tokens','generated_tokens_all_phases','non_loss_generation_tokens',
   'rollouts','binary_verifier_test_calls','challenger_generated_tokens'],
 'slurm_allocations':['job_id','round','search_branch_index','stage','state','elapsed_seconds','allocated_cpus','allocated_gpus',
   'allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours','batch_max_rss_bytes','parent_job_id','version'],
}


def expected_tables(bundle):
    d=bundle['analysis'];validate_analysis(d)
    profiles=[]
    for row in d['endpoint_rows']:
        r=row['round'];label=row['endpoint']
        if label=='branch_endpoint':source=f"raw_results/verge_search_v1_r{r:02d}_b{row['branch_index']}/branches/0/{row['kind']}.jsonl"
        else:
            suffix=f"initial/{row['kind']}" if label=='initial' else 'completion'
            source=f"raw_results/verge_search_v1/rounds/{row['round_version']}/{suffix}.jsonl"
        profiles.append(dict(row,source_artifact=source))
    contrasts=[dict(r,paired_95_lower=r['paired_95_interval'][0],paired_95_upper=r['paired_95_interval'][1],
        bootstrap_replicates=d['bootstrap_replicates']) for r in d['within_search_contrasts']]
    events=[]
    for r in d['trajectory_events']:
        first=r['first_event'] or {}
        events.append(dict(r,first_event_role=first.get('role'),first_event_update=first.get('update'),
            first_event_group=first.get('group_index'),generated_tokens_through_first_group=first.get('generated_tokens_through_group')))
    selections=[dict(r,eligible_branches_json=json.dumps(r['eligible_branches'],separators=(',',':')),
        selection_probabilities_json=None if r['selection_probabilities'] is None else json.dumps(r['selection_probabilities'],separators=(',',':')))
        for r in d['round_selections']]
    return {'condition_profiles':profiles,'paired_contrasts':contrasts,'trajectory_events':events,'round_selections':selections,
        'generation_costs':d['generation_ledger'],'slurm_allocations':[dict(r,batch_max_rss_bytes=r.get('batch_max_rss_bytes'))
            for r in bundle['resources']['allocations']]}


def validate_csv(bundle,directory):
    if type(bundle.get('synthetic')) is not bool:raise RuntimeError('Explicit source provenance required')
    expected=expected_tables(bundle);directory=Path(directory)
    receipt=json.loads((directory/'csv_export_receipt.json').read_text(encoding='utf-8'))
    if (receipt.get('protocol')!='verge_search_v1_csv_export' or receipt.get('synthetic') is not bundle['synthetic']
            or receipt.get('suite_complete') is not False or receipt.get('search_block_complete') is not False):
        raise RuntimeError('Wrong search CSV provenance or completion claim')
    entries={r['name']:r for r in receipt['tables']}
    if set(entries)!=set(COLUMNS) or len(receipt['tables'])!=6:raise RuntimeError('Missing or duplicate CSV receipt')
    for name,columns in COLUMNS.items():
        rows=expected[name];entry=entries[name]
        if (entry.get('rows')!=len(rows) or entry.get('columns')!=len(columns)
                or entry.get('typed_values_unchanged') is not True or entry.get('preview_rendered') is not True
                or entry.get('preview_columns')!=8 or entry.get('formula_error_check_available') is not True):
            raise RuntimeError('CSV authoring or preview check incomplete')
        with (directory/(name+'.csv')).open(newline='',encoding='utf-8') as handle:saved=list(csv.reader(handle))
        if len(saved)!=len(rows)+1 or saved[0]!=columns:raise RuntimeError('CSV row or column set changed')
        for actual,row in zip(saved[1:],rows):
            if len(actual)!=len(columns) or any(not same_cell(a,row[c]) for a,c in zip(actual,columns)):
                raise RuntimeError('CSV cell differs from independently reconstructed source')
    return {'csv_tables_verified':6,'source_values_unchanged':True,'synthetic':bundle['synthetic'],
        'search_block_complete':False,'suite_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle');p.add_argument('directory');a=p.parse_args()
    print(json.dumps(validate_csv(json.loads(Path(a.bundle).read_text(encoding='utf-8')),a.directory),indent=2))
