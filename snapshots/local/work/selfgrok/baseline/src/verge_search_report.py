"""Search report rendering from verified endpoints, events and allocation data.

No new draws, checkpoint choices, final-test access or completion markers.
CSV export and the final independent artifact gate remain separate.
"""
import argparse
from pathlib import Path
import verge_search_report_values as inputs


def summary_rows(bundle):
    data=bundle['analysis'];indexed=inputs.validate_analysis(data)
    allocations=bundle['resources']['allocations']
    ownership=('version','round','search_branch_index','stage','parent_job_id')
    registered={row['job_id']:{k:row[k] for k in ownership} for row in allocations}
    hardware=inputs.resource_totals(bundle['resources'],registered)
    if bundle['hardware']!=hardware:raise ValueError('Saved report hardware totals are stale')
    result=[]
    for r in range(6):
        choice=next(s for s in data['round_selections'] if s['round']==r)
        name=choice['selected_name'];label='initial' if name=='start' else 'branch_endpoint'
        b=None if name=='start' else 0 if name=='direct' else int(name[-1])
        point=lambda kind,metric:indexed[r,label,b,kind,metric]
        post=lambda metric:indexed[r,'selected_completion',None,'target',metric]
        cost=next(c for c in data['generation_ledger'] if c['round']==r)
        stage_gpu={stage:sum(row['gpu_allocation_hours'] for row in allocations
                            if row['round']==r and row['stage']==stage) for stage in ('prepare','worker','finish')}
        result.append({'round':r,'selected_name':name,'selected_checkpoint':choice['selected_checkpoint'],
            'selection_endpoint_target_full_count':point('target','full_pass')['observed_count'],
            'selection_endpoint_target_draws':512,
            'selection_endpoint_scope_full_count':round(128*point('scope','full_pass_rate')['observed_rate']),
            'selection_endpoint_scope_draws':128,
            'selection_endpoint_scope_case_fraction':point('scope','mean_case_fraction')['observed_rate'],
            'fresh_completion_target_full_count':post('full_pass')['observed_count'],
            'fresh_completion_parse_count':post('parse')['observed_count'],'fresh_completion_draws':64,
            'round_loss_tokens_all_branches':cost['loss_tokens'],
            'generated_tokens_all_phases':cost['generated_tokens_all_phases'],
            'gpu_allocation_hours':sum(stage_gpu.values()),'gpu_hours_by_stage':stage_gpu,
            'branch_events_observed':sum(e['event_observed'] for e in data['trajectory_events'] if e['round']==r),
            'branch_events_right_censored':sum(e['right_censored'] for e in data['trajectory_events'] if e['round']==r)})
    return result


def figure(bundle,destination):
    summary=summary_rows(bundle);indexed=inputs.validate_analysis(bundle['analysis'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.8),layout='constrained')
    rounds=list(range(1,7))
    styles=[('Initial',None,'initial','#858d98','--','x'),('Reference',0,'branch_endpoint','#35465c','-','o'),
            ('Branch 1',1,'branch_endpoint','#177da5','--','s'),('Branch 2',2,'branch_endpoint','#d38139',':','^'),
            ('Branch 3',3,'branch_endpoint','#836cb0','-.','d')]
    for ax,metric,title in [(axes[0,0],'full_pass','A  Target full pass: fixed endpoints (N=512)'),
                            (axes[0,1],'parse','B  Target parse: fixed endpoints (N=512)')]:
        all_rates=[]
        for label,b,role,color,line,marker in styles:
            rates=[100*indexed[r,role,b,'target',metric]['observed_rate'] for r in range(6)];all_rates+=rates
            ax.plot(rounds,rates,label=label,color=color,linestyle=line,marker=marker,markersize=4,linewidth=1.4)
        ax.set_title(title,loc='left',fontsize=10);ax.set_ylabel('Observed rate (%)',fontsize=9)
        ax.set_ylim(-.03*max(1.,max(all_rates)),max(1.,1.18*max(all_rates)))
        if not any(all_rates):ax.text(.03,.9,'All displayed curves coincide at zero',transform=ax.transAxes,fontsize=8,color='#505b68')
    ax=axes[1,0]
    full=[100*s['fresh_completion_target_full_count']/64 for s in summary]
    parse=[100*s['fresh_completion_parse_count']/64 for s in summary]
    ax.plot(rounds,full,color='#35465c',marker='o',label='Full target pass',markersize=4)
    ax.plot(rounds,parse,color='#177da5',marker='s',linestyle='--',label='Target parse',markersize=4)
    ax.set_title('C  Selected checkpoint: fresh check (N=64)',loc='left',fontsize=10)
    ax.set_ylabel('Observed rate (%)',fontsize=9);ax.set_ylim(-.03*max(1.,max(full+parse)),max(1.,1.18*max(full+parse)))
    ax.legend(loc='upper left',fontsize=8,frameon=False)
    ax=axes[1,1];bottom=[0.]*6
    for stage,label,color in [('prepare','Shared initial','#91b4c9'),('worker','All four workers','#35465c'),('finish','Selection / fresh check','#d38139')]:
        heights=[s['gpu_hours_by_stage'][stage] for s in summary]
        ax.bar(rounds,heights,bottom=bottom,color=color,label=label,width=.65)
        bottom=[a+b for a,b in zip(bottom,heights)]
    ax.set_title('D  Allocated GPU time, including recovery',loc='left',fontsize=10)
    ax.set_ylabel('Allocated GPU-hours',fontsize=9)
    ax.set_ylim(0,max(1.,max(bottom)*1.45));ax.legend(loc='upper left',fontsize=8,frameon=False)
    for ax in axes.flat:
        ax.set_xticks(rounds);ax.set_xlabel('Search round',fontsize=9);ax.tick_params(labelsize=8)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8,frameon=False)
    fig.suptitle(('SYNTHETIC TEST DATA — ' if bundle['synthetic'] else '')+
                 'Target-only branch search\nOne training seed; endpoint and trajectory observations are not pooled',fontsize=11)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160);plt.close(fig)


def markdown(bundle):
    summary=summary_rows(bundle);data=bundle['analysis']
    lines=['# Target-only branch search','']
    if bundle['synthetic']:lines+=['SYNTHETIC TEST DATA — not model results.','']
    lines+=['Six rounds of four target-only binary-reward branches from a shared round start. '
            'Each branch receives 524,288 loss tokens and a fresh optimizer. '
            'The original endpoint selector chooses continuation; no Challenger generates or learns curricula.','',
        '| Round | Selected action | Selection endpoint: target | Fresh check: target | Fresh check: parse | Selection endpoint: scope |',
        '|---|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['round']+1} | {s['selected_name']} | {s['selection_endpoint_target_full_count']}/512 | "
            f"{s['fresh_completion_target_full_count']}/64 | {s['fresh_completion_parse_count']}/64 | {s['selection_endpoint_scope_full_count']}/128 |")
    lines+=['','The selected policy\'s selection endpoint and its later fresh check are separate batches. '
        'The fresh check never changes that round\'s choice. Scope counts come from the selected policy\'s '
        'pre-selection endpoint, not an additional post-selection scope sample.','',
        '![Search endpoints and allocation costs](search_trajectories.png)','',
        '| Round | Loss tokens: all branches | Generated tokens: all phases | Allocated GPU-hours | Observed branch events | Right-censored branches |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['round']+1} | {s['round_loss_tokens_all_branches']:,} | {s['generated_tokens_all_phases']:,} | "
            f"{s['gpu_allocation_hours']:.4f} | {s['branch_events_observed']} | {s['branch_events_right_censored']} |")
    lines+=['',f"Total: {data['total_loss_tokens']:,} loss tokens; "
        f"{bundle['hardware']['totals']['gpu_allocation_hours']:.4f} allocated GPU-hours and "
        f"{bundle['hardware']['totals']['cpu_allocation_core_hours']:.4f} allocated CPU-core-hours.",'',
        data['event_note'],'',data['statistical_note'],'',
        data.get('cost_note','Unique committed generation files only; shared initial and selected completion count once.'),'',
        'All four worker allocations and shared initial/selection/fresh-check allocations are charged. '
        'Historical failed allocations remain in the ledger. Hardware time is not active utilization or FLOPs. '
        'Primary, invalidated-v1, direct-control and follow-on costs remain separately disclosed.','',
        'Parse and scope success are not full target success. A zero observation or right-censored branch '
        'does not establish an impossible target. No cross-training-seed or end-to-end-compute-matched claim is made. '
        'Official held-out and sealed tests remain unopened.','',
        'This report covers only the target-only search block. CSV export, independent artifact verification '
        'and the remaining experiment-book matrix are still required; rendering does not mark either block or suite complete.','']
    return '\n'.join(lines)


def latex_table(bundle):
    summary=summary_rows(bundle)
    labels={'start':'Start','direct':'Reference','candidate_1':'Branch 1','candidate_2':'Branch 2','candidate_3':'Branch 3'}
    lines=[r'% SYNTHETIC TEST DATA, not model evidence.' if bundle['synthetic'] else r'% Single training trajectory; fixed-checkpoint observations.',
           r'% Selection N=512 and fresh-check N=64 are not pooled; scope is pre-selection N=128.',
           r'\begin{tabular}{rlrrrr}',r'\hline',
           r'Round & Selected & Selection full & Fresh full & Fresh parse & Scope full \\',r'\hline']
    for s in summary:
        lines.append(f"{s['round']+1} & {labels[s['selected_name']]} & {s['selection_endpoint_target_full_count']}/512 & "
            f"{s['fresh_completion_target_full_count']}/64 & {s['fresh_completion_parse_count']}/64 & {s['selection_endpoint_scope_full_count']}/128 "+r'\\')
    return '\n'.join(lines+[r'\hline',r'\end{tabular}',''])


def render(bundle,directory):
    if type(bundle.get('synthetic')) is not bool:raise ValueError('Explicit synthetic/real provenance is required')
    summary=summary_rows(bundle);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    inputs.ctl.write(directory/'report_bundle.json',bundle)
    inputs.ctl.write(directory/'render_values.json',{'rounds':summary,'synthetic':bundle['synthetic'],'suite_complete':False})
    figure(bundle,directory/'search_trajectories.png')
    (directory/'RESULTS_SEARCH.md').write_text(markdown(bundle),encoding='utf-8')
    (directory/'result_table.tex').write_text(latex_table(bundle),encoding='utf-8')
    inputs.ctl.write(directory/'core_render_receipt.json',{'synthetic':bundle['synthetic'],
        'figure_rendered':True,'markdown_rendered':True,'latex_rendered':True,
        'csv_exported':False,'visual_review_complete':False,'search_block_complete':False,'suite_complete':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    bundle=inputs.collect()
    if args.save:render(bundle,inputs.ctl.ROOT)
    print('Search report core checked; CSV, visual review and artifact validation remain; suite_complete=false')
