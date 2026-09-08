"""Read and reconcile control CSV/figure/LaTeX artifacts without hashes or draws."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import verge_control_controller as ctl
import verge_control_report as report


COLUMNS = {
    'condition_profiles':['arm','round','endpoint','checkpoint','kind','metric','observed_rate','rollouts','loss_tokens_before_endpoint','source_artifact'],
    'paired_contrasts':['round','candidate','reference','kind','metric','difference','paired_95_lower','paired_95_upper','bootstrap_replicates'],
    'generation_costs':['arm','round','loss_tokens','nonzero_advantage_tokens','generated_training_tokens',
        'generated_initial_endpoint_tokens','generated_final_endpoint_tokens','generated_completion_tokens_all_phases',
        'non_loss_generation_tokens','raw_rollouts','binary_verifier_test_calls','challenger_generated_tokens'],
    'trajectory_events':['arm','round','updates','optimizer_steps','training_full_successes','first_full_success_observed',
        'first_success_update','first_success_phase','generated_tokens_before_first_success_update','scope_safe_vs_start_observed'],
    'slurm_allocations':['job_id','arm','round','stage','state','elapsed_seconds','allocated_cpus','allocated_gpus',
        'allocated_cpu_memory_bytes','gpu_allocation_hours','cpu_allocation_core_hours','batch_max_rss_bytes'],
}


def expected_tables(bundle):
    data = bundle['analysis']
    profiles = [dict(r,source_artifact=f"raw_results/verge_control_v1_{r['arm']}_r{r['round']:02d}/"
        f"{'initial' if r['endpoint'] == 'initial' else 'branches/0'}/{r['kind']}.jsonl") for r in data['endpoint_rows']]
    contrasts = [dict(r,paired_95_lower=r['paired_95_interval'][0],paired_95_upper=r['paired_95_interval'][1],
        bootstrap_replicates=data['bootstrap_replicates']) for r in data['within_control_contrasts']]
    events = []
    for r in data['trajectory_events']:
        event = r['first_training_full_success']
        events.append(dict(r,first_full_success_observed=event is not None,
            first_success_update=event['update'] if event else None,first_success_phase=event['phase'] if event else None,
            generated_tokens_before_first_success_update=event['generated_tokens_before'] if event else None))
    return {'condition_profiles':profiles,'paired_contrasts':contrasts,'generation_costs':data['generation_ledger'],
            'trajectory_events':events,'slurm_allocations':bundle['resources']['allocations']}


def same_cell(actual, expected):
    if expected is None:
        return actual == ''
    if isinstance(expected,bool):
        return actual == str(expected).lower()
    if isinstance(expected,(int,float)):
        try:
            # The JS and Python runtimes can print the same binary64 number with
            # different shortest decimal tails. Reparse numbers, without rounding
            # scientific outputs or replacing missing numeric fields by zero.
            if actual == '' or not Decimal(actual).is_finite():
                return False
            return Decimal(actual) == Decimal(expected) if isinstance(expected,int) else float(actual) == expected
        except (InvalidOperation,ValueError):
            return False
    return actual == expected


def validate_csv(bundle, directory):
    expected = expected_tables(bundle)
    receipt = ctl.read(directory/'csv_export_receipt.json')
    if (receipt.get('protocol') != 'verge_control_v1_csv_export'
            or receipt.get('synthetic') is not bundle.get('synthetic')
            or receipt.get('suite_complete') is not False):
        raise RuntimeError('Wrong CSV export receipt')
    entries = {r['name']:r for r in receipt['tables']}
    if set(entries) != set(COLUMNS) or len(receipt['tables']) != len(COLUMNS):
        raise RuntimeError('Missing or duplicate table receipt')
    for name, columns in COLUMNS.items():
        rows = expected[name]
        record = entries[name]
        if (record['rows'] != len(rows) or record['columns'] != len(columns)
                or record.get('typed_values_unchanged') is not True or record.get('preview_rendered') is not True):
            raise RuntimeError('CSV export or render not verified')
        with (directory/(name+'.csv')).open(newline='',encoding='utf-8') as handle:
            saved = list(csv.reader(handle))
        if len(saved) != len(rows)+1 or saved[0] != columns:
            raise RuntimeError('CSV row/column set is incomplete')
        for saved_row, row in zip(saved[1:],rows):
            if len(saved_row) != len(columns) or any(not same_cell(actual,row[column]) for actual,column in zip(saved_row,columns)):
                raise RuntimeError('CSV value differs from its independently reconstructed source')
    return {'csv_tables_verified':5,'source_values_unchanged':True}


def validate():
    data,snapshot,hardware = report.collect()
    bundle = ctl.read(ctl.ROOT/'report_bundle.json')
    if (bundle.get('synthetic') is not False or bundle['analysis'] != data or bundle['hardware'] != hardware
            or bundle['resources']['allocations'] != snapshot['allocations']):
        raise RuntimeError('Synthetic, stale or incomplete report bundle')
    if (ctl.read(ctl.ROOT/'endpoint_analysis.json') != data
            or ctl.read(ctl.ROOT/'report_values.json') != hardware
            or ctl.read(ctl.ROOT/'resource_accounting_final.json') != bundle['resources']):
        raise RuntimeError('Standalone report records differ from the validated bundle')
    result = validate_csv(bundle,ctl.ROOT)
    if (ctl.ROOT/'RESULTS_DIRECT.md').read_text(encoding='utf-8') != report.report(data,hardware):
        raise RuntimeError('Markdown report differs from source values')
    if (ctl.ROOT/'result_table.tex').read_text(encoding='utf-8') != report.latex_table(data,hardware):
        raise RuntimeError('LaTeX table differs from source values')
    from PIL import Image
    with Image.open(ctl.ROOT/'direct_control_trajectories.png') as picture:
        if picture.format != 'PNG' or picture.width < 800 or picture.height < 400:
            raise RuntimeError('Missing or invalid main control figure')
        picture.verify()
    # A monitor must actually inspect the real plot and rendered exports. A
    # synthetic renderer test cannot substitute for that final visual review.
    review = ctl.read(ctl.ROOT/'artifact_visual_review.json')
    if (review.get('synthetic') is not False or review.get('main_figure_reviewed') is not True
            or set(review.get('csv_previews_reviewed',[])) != set(COLUMNS)
            or review.get('all_labels_and_values_readable') is not True):
        raise RuntimeError('Final real-artifact visual review is missing')
    return dict(result,protocol='verge_control_v1_artifact_validation',control_block_complete=True,
                suite_complete=False,scope='Only binary, dense and dense-to-binary direct controls',
                segments=18,official_test_opened=False,checkpoint_hash_scans=0,new_model_samples=0,
                required_matrix_remaining=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = validate()
    if args.save:
        ctl.write(ctl.ROOT/'DIRECT_CONTROL_BLOCK_COMPLETE.json',result)
    print(ctl.json.dumps(result,indent=2))
