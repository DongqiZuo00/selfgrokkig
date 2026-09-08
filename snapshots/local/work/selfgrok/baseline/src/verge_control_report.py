"""Direct-control report core: verified JSON, fixed-endpoint figure and Markdown.

No sampling or selection occurs here. CSV/LaTeX exports and final artifact
validation are separate; writing this report is never book completion.
"""
import argparse
import math
from pathlib import Path
import verge_control_controller as ctl
import verge_control_analysis as analysis
import verge_control_resources as resources


def resource_totals(snapshot, registered):
    if (snapshot.get('accounting_complete') is not True
            or snapshot.get('missing_individual_accounting_rows')
            or not registered):
        raise RuntimeError('Wait for complete Slurm records, including the final coordinator')
    allocations = snapshot['allocations']
    by_id = {r['job_id']:r for r in allocations}
    if len(by_id) != len(allocations) or set(by_id) != set(registered):
        raise RuntimeError('Missing, duplicate or foreign control allocation')
    arm_totals = {label:{'gpu_allocation_hours':0.,'cpu_allocation_core_hours':0.} for label in ctl.LABELS}
    coordinator = {'gpu_allocation_hours':0.,'cpu_allocation_core_hours':0.}
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    for identity, metadata in registered.items():
        row = by_id[identity]
        if any(row[k] != metadata[k] for k in ('arm','round','stage')):
            raise RuntimeError('Slurm receipt has wrong control ownership')
        if row['state'].split()[0] not in terminal:
            raise RuntimeError('Live allocation cannot be assigned a final cost')
        gpus,cpus,seconds = (row[k] for k in ('allocated_gpus','allocated_cpus','elapsed_seconds'))
        if any(type(x) is not int or x < 0 for x in (gpus,cpus,seconds)):
            raise RuntimeError('Invalid resource units')
        if gpus > (1 if metadata['stage'] == 'worker' else 0):
            raise RuntimeError('Control GPU allocation exceeds its role')
        target = coordinator if metadata['stage'] == 'coordinator' else arm_totals[metadata['arm']]
        expected = {'gpu_allocation_hours':gpus*seconds/3600.,'cpu_allocation_core_hours':cpus*seconds/3600.}
        for metric,value in expected.items():
            if not math.isclose(row[metric],value,abs_tol=1e-10):
                raise RuntimeError('Resource receipt does not reconcile with CPU/GPU seconds')
            target[metric] += value
    for metric in coordinator:
        expected = coordinator[metric]+sum(a[metric] for a in arm_totals.values())
        if not math.isclose(snapshot[metric],expected,abs_tol=1e-10):
            raise RuntimeError('Resource total does not reconcile')
    return {'arms':arm_totals,'shared_cpu_coordination':coordinator,
            'historical_failed_allocations_included':True,
            'note':'Allocated device/core time, not utilization, FLOPs or matched end-to-end compute'}


def validate_analysis(data):
    if (data.get('protocol') != 'verge_control_v1_endpoint_analysis' or data.get('segments') != 18
            or data.get('official_test_opened') is not False or data.get('suite_complete') is not False
            or data.get('total_compute_matched') is not False):
        raise RuntimeError('Not the isolated fixed-endpoint control analysis')
    metrics = [('target',m) for m in analysis.condition_names()]+[('scope',m) for m in ('full_pass_rate','mean_case_fraction')]
    expected = {(arm,r,endpoint,kind,metric) for arm in ctl.LABELS for r in range(6)
                for endpoint in ('initial','final') for kind,metric in metrics}
    indexed = {}
    for row in data['endpoint_rows']:
        key = tuple(row[k] for k in ('arm','round','endpoint','kind','metric'))
        rate = row['observed_rate']
        if key in indexed or key not in expected or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise RuntimeError('Missing, duplicate or invalid endpoint metric')
        expected_tokens = 524288*(row['round']+(row['endpoint'] == 'final'))
        if row['loss_tokens_before_endpoint'] != expected_tokens or row['rollouts'] != (512 if row['kind'] == 'target' else 128):
            raise RuntimeError('Endpoint denominator or loss-token coordinate changed')
        indexed[key] = row
    if set(indexed) != expected:
        raise RuntimeError('Incomplete endpoint matrix; missing data must not become zeros')
    for group in ('generation_ledger','trajectory_events'):
        entries = data[group]
        keys = [(r['arm'],r['round']) for r in entries]
        if len(keys) != 18 or set(keys) != {(a,r) for a in ctl.LABELS for r in range(6)}:
            raise RuntimeError('Incomplete control trajectory or generation ledger')
    if any(c['loss_tokens'] != 524288 for c in data['generation_ledger']):
        raise RuntimeError('Control loss budgets do not match the submitted plan')
    return indexed


def figure(data, destination, *, synthetic=False):
    indexed = validate_analysis(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('target','full_pass','Target full pass'),('target','parse','Target parse'),
              ('scope','full_pass_rate','Scope full pass'),('scope','mean_case_fraction','Scope test-case fraction')]
    styles = [('Binary','#35465c','o','-'),('Dense','#176d9c','s','--'),('Dense to binary','#bc622b','^',':')]
    fig,axes = plt.subplots(2,2,figsize=(9.5,6.1),layout='constrained')
    for ax,(kind,metric,title) in zip(axes.flat,panels):
        all_rates = []
        for arm,(label,color,marker,linestyle) in zip(ctl.LABELS,styles):
            curve = [indexed[arm,0,'initial',kind,metric]]+[indexed[arm,r,'final',kind,metric] for r in range(6)]
            x = [p['loss_tokens_before_endpoint']/1e6 for p in curve]
            y = [100*p['observed_rate'] for p in curve]
            all_rates.extend(y)
            ax.plot(x,y,label=label,color=color,marker=marker,linestyle=linestyle,linewidth=1.6,markersize=4)
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_xlabel('Cumulative loss tokens (millions)',fontsize=9)
        ax.set_ylabel('Observed rate (%)',fontsize=9)
        ax.set_ylim(-.025*max(1.,max(all_rates)),max(1.,max(all_rates)*1.12))
        ax.tick_params(labelsize=8)
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        if not any(all_rates):
            ax.text(.03,.87,'All displayed observations are zero',transform=ax.transAxes,fontsize=8,color='#505b68')
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(0.,.79 if not any(
        p['observed_rate'] for p in data['endpoint_rows'] if p['metric'] == 'full_pass') else 1.),fontsize=8,frameon=False)
    fig.suptitle(('SYNTHETIC TEST DATA — ' if synthetic else '')+'Direct training controls\nFixed endpoints, one training seed; no cross-seed error bands',fontsize=11)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def report(data, hardware):
    indexed = validate_analysis(data)
    warmup = data['dense_warmup_loss_tokens']
    fraction = warmup/(6*524288)
    lines = ['# Direct training controls','',
        'Three target-only arms, each with six fixed 524,288-loss-token segments. '
        'The endpoint at the end of segment six is reported without checkpoint selection.','',
        '| Arm | Final target full pass | Final parse | Final scope full pass | Loss tokens | Generated tokens, all phases | Allocated GPU-hours |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ctl.LABELS:
        full = indexed[arm,5,'final','target','full_pass']['observed_rate']
        parse = indexed[arm,5,'final','target','parse']['observed_rate']
        scope = indexed[arm,5,'final','scope','full_pass_rate']['observed_rate']
        costs = [c for c in data['generation_ledger'] if c['arm'] == arm]
        gpu = hardware['arms'][arm]['gpu_allocation_hours']
        lines.append(f"| {arm} | {round(full*512)}/512 | {round(parse*512)}/512 | {round(scope*128)}/128 | "
            f"{sum(c['loss_tokens'] for c in costs):,} | {sum(c['generated_completion_tokens_all_phases'] for c in costs):,} | {gpu:.4f} |")
    lines += ['',
        f'The dense-to-binary arm uses dense reward for the first {warmup:,} loss tokens '
        f'({fraction:.2%} of its total budget), then binary reward. All arms reset their optimizer '
        'at the same segment boundaries; the switch within segment two preserves optimizer state.',
        '', '![Fixed-endpoint trajectories](direct_control_trajectories.png)', '',
        'Each curve includes the initial segment-one endpoint and the six final endpoints. '
        'Fresh initial measurements at later segment starts remain in the underlying analysis '
        'but are not inserted as extra training progress points.', '',
        '## Interpretation and accounting','',data['statistical_note'],'',data['cost_note'],'',
        'Training first-success events are separate from endpoint pass rates. '
        'Parse or test-case reward is never counted as a complete target success.', '',
        f"Shared CPU coordination: {hardware['shared_cpu_coordination']['cpu_allocation_core_hours']:.4f} allocated CPU-core-hours. "
        'Failed historical allocations remain charged to their registered arm. '
        'The Mistral primary block and invalidated v1 costs are recorded separately and are not included in this control table.', '',
        'These three direct arms are not the four-branch target-only search control or '
        'an end-to-end-compute-matched comparison. No cross-primary paired contrast is made. '
        'Official held-out and sealed tests remain unopened.', '',
        'This report covers only these three direct controls. The other required experiment-book blocks remain outstanding. '
        'This report does not mark the control block or the experiment book complete; completion is recorded only after the separate artifact validation.','']
    return '\n'.join(lines)


def collect():
    if ctl.os.environ.get('VERGE_BOOK_SUITE') != 'verge_book_v2':
        raise RuntimeError('Explicit v2 suite is required')
    data = analysis.collect()
    snapshot = resources.collect()
    registered,_ = resources.registry(ctl.ROOT)
    hardware = resource_totals(snapshot,registered)
    # Recovery predecessors may fail, but each *current* registered wave must
    # have completed. A partial/cancelled last coordinator is not finalized.
    states = {r['job_id']:r['state'] for r in snapshot['allocations']}
    for r in range(6):
        jobs = ctl.read(ctl.ROOT / 'jobs' / f'round_{r:02d}.json')
        current = [f"{jobs['workers']}_{i}" for i in range(3)]+[jobs['coordinator']]
        if any(states.get(job) != 'COMPLETED' for job in current):
            raise RuntimeError('Current control wave is not complete')
    return data,snapshot,hardware


def latex_table(data, hardware):
    """Exact fixed-segment counts, not a best-checkpoint or pooled estimate."""
    indexed = validate_analysis(data)
    labels = {'binary':'Binary','dense':'Dense','dense_then_binary':'Dense to binary'}
    lines = [r'% Fixed final endpoints, single training seed. Counts are full task passes.',
             r'% GPU-hours include registered recovery predecessors; CPU coordination is separate.',
             r'\begin{tabular}{lrrrrr}',r'\hline',
             r'Arm & Target & Parse & Scope & Loss tokens & GPU-hours \\',r'\hline']
    for arm in ctl.LABELS:
        full = round(512*indexed[arm,5,'final','target','full_pass']['observed_rate'])
        parse = round(512*indexed[arm,5,'final','target','parse']['observed_rate'])
        scope = round(128*indexed[arm,5,'final','scope','full_pass_rate']['observed_rate'])
        tokens = sum(c['loss_tokens'] for c in data['generation_ledger'] if c['arm'] == arm)
        gpu = hardware['arms'][arm]['gpu_allocation_hours']
        lines.append(f'{labels[arm]} & {full}/512 & {parse}/512 & {scope}/128 & {tokens:,} & {gpu:.4f} '+r'\\')
    return '\n'.join(lines+[r'\hline',r'\end{tabular}',''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    data,snapshot,hardware = collect()
    if args.save:
        ctl.write(ctl.ROOT / 'endpoint_analysis.json',data)
        ctl.write(ctl.ROOT / 'resource_accounting_final.json',snapshot)
        ctl.write(ctl.ROOT / 'report_values.json',hardware)
        ctl.write(ctl.ROOT / 'report_bundle.json',{'analysis':data,'resources':snapshot,'hardware':hardware,'synthetic':False})
        figure(data,ctl.ROOT / 'direct_control_trajectories.png')
        (ctl.ROOT / 'RESULTS_DIRECT.md').write_text(report(data,hardware),encoding='utf-8')
        (ctl.ROOT / 'result_table.tex').write_text(latex_table(data,hardware),encoding='utf-8')
    print('Direct-control report core checked; control_block_complete=false; suite_complete=false')
