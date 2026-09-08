"""Verified all-candidate report core; never selects checkpoints or opens tests.

The fixed-endpoint chart and censored-event rankings describe different objects.
Writing these artifacts does not finish this block or the experiment book.
"""
import argparse
import math
import verge_followon_controller as ctl
import verge_followon_analysis as analysis
import verge_followon_resources as resources
from verge_followon_protocol import ARMS
from verge_followon_ranks import summarize_cohort


def validate_analysis(data):
    if (data.get('protocol')!='verge_followon_v1_all_candidate_analysis'
            or data.get('candidate_count')!=72 or data.get('cohort_count')!=24
            or data.get('official_test_opened') is not False or data.get('suite_complete') is not False
            or data.get('followon_block_complete') is not False or data.get('total_compute_matched') is not False):
        raise ValueError('Not a complete all-candidate analysis input')
    events={r['array_index']:r for r in data['trajectory_events']}
    if len(data['trajectory_events'])!=72 or set(events)!=set(range(72)) or len(data['cohorts'])!=24:
        raise ValueError('Missing or duplicate follow-on event record')
    indexed={}
    names=analysis.conditions()
    expected={(i,label,kind,metric) for i in range(72)
              for label in ('source','final')
              for kind,metrics in (('target',names),('scope',('full_pass_rate','mean_case_fraction')))
              for metric in metrics}
    expected|={(i,'final_completion','completion',metric) for i in range(72) for metric in names}
    for row in data['endpoint_rows']:
        key=tuple(row[k] for k in ('array_index','endpoint','kind','metric'))
        if key not in expected or key in indexed:raise ValueError('Missing, duplicate or foreign endpoint coordinate')
        i,label,kind,metric=key;e=events[i];n=128 if kind=='scope' else 64 if kind=='completion' else 512
        expected_checkpoint=e['source_checkpoint'] if label=='source' else e['final_checkpoint']
        if (any(row[k]!=e[k] for k in ('version','source_cohort','source_arm','source_round',
                                      'candidate_index','source_solver_initial','pre_success_analysis'))
                or row['checkpoint']!=expected_checkpoint or row['rollouts']!=n
                or row['followon_loss_tokens']!=(0 if label=='source' else 262144)
                or row['source_endpoint_reused'] is not (label=='source')):
            raise ValueError('Mixed-source checkpoint, endpoint denominator or token coordinate')
        value=row['observed_rate'];count=row['observed_count']
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid endpoint rate')
        if kind=='scope':
            if count is not None:raise ValueError('Do not fabricate an integer count for a fractional scope statistic')
        elif type(count) is not int or not 0<=count<=n or not math.isclose(value,count/n,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Endpoint count and rate disagree')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('Incomplete endpoint matrix; missing is not zero')
    rebuilt=[]
    for number in range(24):
        group=[events[i] for i in range(3*number,3*number+3)]
        first=group[0];arm=ARMS[number%4];r=number//4;name=f'verge_book_v2_{arm}_r{r:02d}'
        for j,e in enumerate(group):
            if (e['version']!=ctl.version(3*number+j) or e['source_cohort']!=name
                    or e['source_arm']!=arm or e['source_round']!=r or e['candidate_index']!=j+1
                    or e['source_solver_initial']!=first['source_solver_initial']
                    or type(e['pre_success_analysis']) is not bool
                    or e['pre_success_analysis']!=first['pre_success_analysis']
                    or e['right_censored'] is not (not e['event_observed'])):
                raise ValueError('Foreign or inconsistent source cohort/event')
            for label,kind,field in (('final','target','target_endpoint_full_successes'),
                                     ('final_completion','completion','completion_full_successes')):
                if indexed[e['array_index'],label,kind,'full_pass']['observed_count']!=e[field]:
                    raise ValueError('Trajectory metadata substituted for a fixed endpoint')
            if e['event_observed'] != bool(e['target_training_full_successes'] or e['target_endpoint_full_successes'] or e['completion_full_successes']):
                raise ValueError('Follow-on event flag differs from its observation roles')
            for label,kind in (('source','target'),('final','target'),('final_completion','completion')):
                counts=[indexed[e['array_index'],label,kind,m]['observed_count'] for m in names]
                if any(b>a for a,b in zip(counts,counts[1:])):raise ValueError('Non-nested endpoint ladder')
        source={'cohort':name,'source_solver_initial':first['source_solver_initial'],'candidates':group}
        summary=summarize_cohort(source,group)
        summary.update(source_arm=arm,source_round=r,pre_success_analysis=first['pre_success_analysis'])
        saved=data['cohorts'][number]
        if any(saved[k]!=v for k,v in summary.items() if k!='analysis_plan_frozen'):
            raise ValueError('Saved cohort ranking differs from the retained scores and censoring')
        rebuilt.append(summary)
    if len(data['cohorts'])!=24 or data['descriptive_aggregates']!=analysis.descriptive_aggregates(rebuilt):
        raise ValueError('Descriptive aggregate differs from its within-cohort comparisons')
    costs=data['generation_ledger']
    if len(costs)!=72 or {c['array_index'] for c in costs}!=set(range(72)):
        raise ValueError('Incomplete committed generation ledger')
    for c in costs:
        fields=('generated_training_tokens','generated_target_endpoint_tokens','generated_scope_endpoint_tokens','generated_completion_tokens')
        if (c['version']!=ctl.version(c['array_index']) or c['loss_tokens']!=262144
                or c['generated_initial_endpoint_tokens']!=0
                or any(type(c[k]) is not int or c[k]<0 for k in fields)
                or c['generated_training_tokens']<262144
                or c['generated_completion_tokens_all_phases']!=sum(c[k] for k in fields)
                or c['non_loss_generation_tokens']!=sum(c[k] for k in fields)-262144):
            raise ValueError('Generation budget or reused-source accounting differs')
    return indexed


def resource_totals(snapshot,registered):
    if (not registered or snapshot.get('accounting_complete') is not True
            or snapshot.get('missing_individual_accounting_rows')):
        raise ValueError('Wait for all registered allocation records to be terminal')
    rows=snapshot['allocations'];by_id={r['job_id']:r for r in rows}
    if len(rows)!=len(by_id) or set(by_id)!=set(registered):
        raise ValueError('Missing, duplicate or foreign Slurm allocation')
    totals={a:{'gpu_allocation_hours':0.,'cpu_allocation_core_hours':0.} for a in ARMS}
    shared={'gpu_allocation_hours':0.,'cpu_allocation_core_hours':0.}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    for identity,metadata in registered.items():
        row=by_id[identity]
        if any(row.get(k)!=v for k,v in metadata.items()) or row['state'].split()[0] not in terminal:
            raise ValueError('Allocation ownership or terminal state differs')
        worker=metadata['stage']=='worker'
        if metadata['stage'] not in ('worker','coordinator'):raise ValueError('Unknown allocation role')
        gpus,cpus,seconds=(row[k] for k in ('allocated_gpus','allocated_cpus','elapsed_seconds'))
        memory=row.get('allocated_cpu_memory_bytes')
        if (any(type(v) is not int or v<0 for v in (gpus,cpus,seconds))
                or gpus>(1 if worker else 0) or cpus>(8 if worker else 2)
                or (memory is None and cpus>0)
                or (memory is not None and (not math.isfinite(memory) or not 0<=memory<=(32 if worker else 4)*1024**3))):
            raise ValueError('Allocation exceeds its role or has unknown resource units')
        if worker:
            index=metadata['candidate_array_index']
            if metadata['version']!=ctl.version(index):raise ValueError('Worker assigned to a different candidate')
            target=totals[ARMS[index%12//3]]
        else:target=shared
        for metric,value in (('gpu_allocation_hours',gpus*seconds/3600),('cpu_allocation_core_hours',cpus*seconds/3600)):
            if not math.isclose(row[metric],value,rel_tol=0,abs_tol=1e-10):raise ValueError('Allocation seconds do not reconcile')
            target[metric]+=value
    for metric in shared:
        if not math.isclose(shared[metric]+sum(t[metric] for t in totals.values()),snapshot[metric],rel_tol=0,abs_tol=1e-9):
            raise ValueError('Final resource total does not reconcile')
    return {'source_arms':totals,'shared_cpu_coordination':shared,'historical_failed_allocations_included':True,
            'note':'Allocated device/core time, not utilization or FLOPs; primary v1/v2 and direct-control costs stay separate.'}


def figure(data,destination,*,synthetic=False):
    indexed=validate_analysis(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
    selected=[r for r in data['descriptive_aggregates'] if r['stratum']=='pre_success' and r['source_arm']=='all_arms']
    strategies=(('condition','Condition gain','#176d9c','o'),('uncertainty','Uncertainty','#bd6732','s'),('random','Uniform order','#666e78','^'))
    for ax,field,title in ((axes[0,0],'mean_cohort_top_k_event_fraction','A  Observed event fraction among top k'),
                           (axes[0,1],'mean_cohort_top_k_any_event_indicator','B  Any observed event among top k')):
        for key,label,color,marker in strategies:
            points=sorted([r for r in selected if r['strategy']==key],key=lambda r:r['top_k'])
            if all(r[field] is not None for r in points):
                ax.plot([1,2,3],[100*r[field] for r in points],label=label,color=color,marker=marker,linewidth=1.5)
        ax.set(title=title,xlabel='Candidates followed in ranking order',ylabel='Descriptive mean (%)',xticks=[1,2,3],ylim=(-3,103))
        if not selected[0]['cohorts']:ax.text(.5,.5,'No pre-success cohorts',ha='center',transform=ax.transAxes)
    if selected[0]['cohorts']:axes[0,0].legend(fontsize=8,frameon=False)
    ax=axes[1,0]
    for i,(key,label,color,_) in enumerate(strategies):
        row=next(r for r in selected if r['strategy']==key and r['top_k']==1)
        value=row['pair_weighted_descriptive_concordance']
        if value is not None:ax.bar(i,value,color=color,width=.6)
        else:ax.text(i,.07,'N/A',ha='center',fontsize=9)
    pairs=selected[0]['within_cohort_comparable_pairs']
    ax.set(title=f'C  Within-cohort concordance ({pairs} comparable pairs)',ylabel='Concordance',ylim=(0,1.12),
           xlim=(-.6,2.6),xticks=[0,1,2],xticklabels=['Condition','Uncertainty','Uniform'])
    if not pairs:ax.text(.5,.65,'No comparable event times\nN/A does not mean zero concordance',ha='center',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1];values=[]
    for label,name,color,offset,marker in (('source','Source endpoint','#bd6732',-.1,'x'),('final','Final endpoint','#176d9c',.1,'o')):
        y=[100*indexed[i,label,'target','full_pass']['observed_rate'] for i in range(72)];values+=y
        ax.scatter([i+offset for i in range(72)],y,s=14,label=name,color=color,marker=marker,alpha=.75)
    ax.set(title='D  Full-pass rates at fixed candidate checkpoints',xlabel='Source round (12 candidates per round)',
           ylabel='Observed full pass (%)',xticks=[5.5+12*r for r in range(6)],xticklabels=[str(r+1) for r in range(6)],
           ylim=(-.03*max(1.,max(values)),max(1.,max(values)*1.15)))
    ax.legend(fontsize=8,frameon=False)
    if not any(values):ax.text(.5,.5,'All displayed endpoint counts are zero',ha='center',transform=ax.transAxes,fontsize=8)
    for ax in axes.flat:
        ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.xaxis.label.set_fontsize(9);ax.yaxis.label.set_fontsize(9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.suptitle(('SYNTHETIC TEST DATA — ' if synthetic else '')+'All-candidate target-only follow-ons\nA–C: pre-success cohorts, exact ranking ties; D: all candidates; no iid confidence bands',fontsize=11)
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=160);plt.close(fig)


def report(data,hardware,*,synthetic=False):
    validate_analysis(data)
    lines=['# '+('SYNTHETIC TEST DATA — ' if synthetic else '')+'All-candidate follow-ons','',
        'All 72 original candidates receive 262,144 additional target-only binary-reward loss tokens each. '
        'Unselected, scope-ineligible and zero-update candidates are retained. There is no new initial model evaluation, '
        'early stopping or adaptive checkpoint selection.','',
        '| Source arm | Candidates | Pre-success candidates | New observed events | Budget-censored | Candidate loss tokens | Committed generated tokens | Allocated GPU-hours |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        events=[e for e in data['trajectory_events'] if e['source_arm']==arm]
        indices={e['array_index'] for e in events};costs=[c for c in data['generation_ledger'] if c['array_index'] in indices]
        lines.append(f"| {arm} | {len(events)} | {sum(e['pre_success_analysis'] for e in events)} | "
            f"{sum(e['event_observed'] for e in events)} | {sum(e['right_censored'] for e in events)} | "
            f"{sum(c['loss_tokens'] for c in costs):,} | {sum(c['generated_completion_tokens_all_phases'] for c in costs):,} | "
            f"{hardware['source_arms'][arm]['gpu_allocation_hours']:.4f} |")
    lines+=['','![All-candidate follow-on summaries](followon_summary.png)','',
        '## What these numbers measure','',data['endpoint_note'],'',data['event_note'],'',data['aggregate_note'],'',
        'Ranking ties are averaged over compatible orders. The uniform reference enumerates six orders of the same '
        'three executed candidates, not six new training trials. A missing concordance means there were no comparable '
        'event times; it is neither a zero score nor evidence that eventual learning is impossible.','',
        'The source endpoint is reused as metadata and not charged again. Committed generation counts include training, '
        'final target and scope evaluations, and the final completion batch exactly once. They exclude uncommitted '
        'failed-attempt generation; registered failed/recovery allocations remain in the separate Slurm cost ledger. '
        'Equal loss-token budgets do not imply equal total compute.','',
        f"Shared CPU coordination: {hardware['shared_cpu_coordination']['cpu_allocation_core_hours']:.4f} allocated CPU-core-hours. "
        'Primary v1/v2 and direct-control allocations are separate and must also be disclosed in the full experiment-book total.','',
        'These are one-training-seed observations. Condition gains are not complete target successes, '
        'and these descriptive rankings alone do not establish learned Challenger advantage or cross-start transfer. '
        'Official held-out and sealed tests remain unopened.','',
        'This report does not mark the follow-on block or experiment book complete. CSV/LaTeX exports, '
        'visual review and a separate artifact-completion validator are still required.','']
    return '\n'.join(lines)


def collect():
    if ctl.os.environ.get('VERGE_BOOK_SUITE')!='verge_book_v2' or ctl.os.environ.get('VERGE_FOLLOWON_SUITE')!='verge_followon_v1':
        raise RuntimeError('Explicit source and follow-on suites are required')
    data=analysis.collect();validate_analysis(data)
    snapshot=resources.collect();registered,_=resources.registry(ctl.ROOT)
    hardware=resource_totals(snapshot,registered)
    jobs=ctl.read(ctl.ROOT/'jobs/block.json');states={r['job_id']:r['state'] for r in snapshot['allocations']}
    current=([f"{jobs['workers']}_{i}" for i in jobs.get('worker_indices',jobs['indices'])]
             if jobs.get('workers') else [])+[jobs['coordinator']]
    if any(states.get(job)!='COMPLETED' for job in current):
        raise RuntimeError('Current worker array and final coordinator must exit successfully')
    if data.get('synthetic') is not False or data.get('analysis_plan_frozen') is not True:
        raise RuntimeError('Synthetic or unfrozen analysis cannot be a real report')
    return data,snapshot,hardware


def latex_tables(data,hardware,*,synthetic=False):
    """Exact event counts and descriptive ranks; never pool endpoint policies."""
    validate_analysis(data)
    labels={'verge':'VERGE','frozen':'Frozen','uncertainty':'Uncertainty','outcome':'Outcome'}
    lines=[r'% '+('SYNTHETIC TEST DATA. ' if synthetic else '')+'Single training seed; all 72 candidates retained.',
        r'% New events are observed anywhere in the follow-on, not final-checkpoint pass counts.',
        r'% Missing concordance is N/A; no comparable event-time pairs is not a zero score.',
        r'\begin{tabular}{lrrrrr}',r'\hline',
        r'Source arm & Candidates & Pre-success & Events & Censored & GPU-hours \\',r'\hline']
    for arm in ARMS:
        events=[e for e in data['trajectory_events'] if e['source_arm']==arm]
        lines.append(f"{labels[arm]} & {len(events)} & {sum(e['pre_success_analysis'] for e in events)} & "
            f"{sum(e['event_observed'] for e in events)} & {sum(e['right_censored'] for e in events)} & "
            f"{hardware['source_arms'][arm]['gpu_allocation_hours']:.4f} "+r'\\')
    lines += [r'\hline',r'\end{tabular}','',
        r'% Each candidate receives 262,144 loss tokens; total 18,874,368. Equal loss tokens are not equal compute.',
        r'% GPU-hours include registered recovery allocations; shared CPU coordination is reported separately.',
        r'% Ranking summaries below use pre-success cohorts only, with exact averaging of ranking ties.',
        r'\begin{tabular}{lrrrrr}',r'\hline',
        r'Ranking & Cohorts & Top-1 & Top-2 & Top-3 & Concordance \\',r'\hline']
    def display(value):return 'N/A' if value is None else f'{value:.4f}'
    for strategy,label in (('condition','Condition gain'),('uncertainty','Uncertainty'),('random','Uniform order')):
        rows=sorted([r for r in data['descriptive_aggregates'] if r['stratum']=='pre_success'
            and r['source_arm']=='all_arms' and r['strategy']==strategy],key=lambda r:r['top_k'])
        values=' & '.join(display(r['mean_cohort_top_k_event_fraction']) for r in rows)
        lines.append(f"{label} & {rows[0]['cohorts']} & {values} & "
            f"{display(rows[0]['pair_weighted_descriptive_concordance'])} "+r'\\')
    return '\n'.join(lines+[r'\hline',r'\end{tabular}','',
        r'% Top-k entries are means of within-cohort observed event fractions, not iid success probabilities.',
        r'% The uniform reference enumerates six orders of the same three candidates, not six training trials.',
        r'% This table does not establish cross-start robustness or complete the experiment book.',''])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    data,snapshot,hardware=collect()
    if args.save:
        ctl.write(ctl.ROOT/'analysis.json',data);ctl.write(ctl.ROOT/'resource_accounting_final.json',snapshot)
        ctl.write(ctl.ROOT/'report_values.json',hardware)
        ctl.write(ctl.ROOT/'report_bundle.json',{'analysis':data,'resources':snapshot,'hardware':hardware,'synthetic':False})
        figure(data,ctl.ROOT/'followon_summary.png')
        (ctl.ROOT/'RESULTS_FOLLOWON.md').write_text(report(data,hardware),encoding='utf-8')
        (ctl.ROOT/'result_tables.tex').write_text(latex_tables(data,hardware),encoding='utf-8')
    print('Follow-on report core checked; followon_block_complete=false; suite_complete=false')
