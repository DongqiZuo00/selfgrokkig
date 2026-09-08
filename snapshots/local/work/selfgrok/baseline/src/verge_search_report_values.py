"""Validate search report inputs and reconcile allocation costs before rendering.

This is not a completion gate. It samples nothing and writes no scientific
conclusion; a later artifact pipeline must render and verify the real outputs.
"""
import argparse
import math
import os
import verge_search_controller as ctl
import verge_search_analysis as analysis
import verge_search_resources as resources
from verge_search_protocol import ROOT_CHECKPOINT
from verge_round_core import condition_names


def validate_analysis(data):
    required={'protocol':'verge_search_v1_endpoint_analysis','rounds':6,'branches':24,
              'total_loss_tokens':12582912,'search_block_complete':False,'suite_complete':False,
              'official_test_opened':False,'new_model_samples':0,'total_compute_matched':False,
              'cross_primary_paired_comparisons':False,'bootstrap_replicates':2000}
    if any(data.get(k)!=v or type(data.get(k)) is not type(v) for k,v in required.items()):
        raise ValueError('Not an isolated six-round search analysis')
    names=condition_names();scope=('full_pass_rate','mean_case_fraction')
    expected={(r,label,b,kind,m) for r in range(6)
              for label,b in [('initial',None)]+[('branch_endpoint',i) for i in range(4)]
              for kind,metrics in [('target',names),('scope',scope)] for m in metrics}
    expected|={(r,'selected_completion',None,'target',m) for r in range(6) for m in names}
    indexed={}
    for row in data['endpoint_rows']:
        key=tuple(row[k] for k in ('round','endpoint','branch_index','kind','metric'))
        if key not in expected or key in indexed:raise ValueError('Duplicate or foreign endpoint coordinate')
        r,label,b,kind,_=key;n=128 if kind=='scope' else 64 if label=='selected_completion' else 512
        tokens=0 if label=='initial' else 524288 if label=='branch_endpoint' else None
        value=row['observed_rate'];count=row['observed_count']
        if (row['rollouts']!=n or row['branch_training_loss_tokens']!=tokens
                or row['round_version']!=ctl.version(r) or not isinstance(row['checkpoint'],str)
                or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
            raise ValueError('Endpoint denominator, checkpoint or branch-local coordinate changed')
        if kind=='scope':
            if count is not None:raise ValueError('Fractional scope metric must not acquire a fabricated count')
            if key[-1]=='full_pass_rate' and not math.isclose(value*n,round(value*n),rel_tol=0,abs_tol=1e-10):
                raise ValueError('Scope full-pass rate is not an integer count over its stated draws')
        elif (type(count) is not int or not 0<=count<=n
              or not math.isclose(value,count/n,rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Endpoint count and rate differ')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('Missing endpoints cannot become zero observations')
    events={(e['round'],e['branch_index']):e for e in data['trajectory_events']}
    if len(data['trajectory_events'])!=24 or set(events)!={(r,b) for r in range(6) for b in range(4)}:
        raise ValueError('All 24 separate branch event records are required')
    selected={s['round']:s for s in data['round_selections']}
    if len(data['round_selections'])!=6 or set(selected)!=set(range(6)):
        raise ValueError('All six original selections are required')
    previous=ROOT_CHECKPOINT
    for r in range(6):
        start=indexed[r,'initial',None,'target','full_pass']['checkpoint']
        if start!=previous:raise ValueError('Selected-checkpoint lineage was changed')
        for label,b in [('initial',None)]+[('branch_endpoint',i) for i in range(4)]+[('selected_completion',None)]:
            rows=[indexed[r,label,b,'target',m] for m in names]
            if len({p['checkpoint'] for p in rows})!=1 or any(y['observed_count']>x['observed_count'] for x,y in zip(rows,rows[1:])):
                raise ValueError('Mixed-checkpoint or non-nested condition ladder')
            if label!='selected_completion' and any(indexed[r,label,b,'scope',m]['checkpoint']!=rows[0]['checkpoint'] for m in scope):
                raise ValueError('Scope belongs to another policy')
        for b in range(4):
            e=events[r,b];endpoint=indexed[r,'branch_endpoint',b,'target','full_pass']
            first=e['first_event'];seen=bool(e['target_training_full_successes'] or e['target_endpoint_full_successes'])
            if (e['version']!=ctl.version(r,b) or e['search_round']!=r
                    or e['checkpoint']!=endpoint['checkpoint'] or e['planned_loss_tokens']!=524288
                    or e['executed_loss_tokens']!=524288 or e['budget_complete'] is not True
                    or e['target_endpoint_full_successes']!=endpoint['observed_count']
                    or e['event_observed'] is not seen or e['right_censored'] is not (not seen)
                    or (first is None)!= (not seen)
                    or e['observation_loss_tokens']!=(first['observation_loss_tokens'] if seen else 524288)
                    or not 0<=e['observation_loss_tokens']<=524288
                    or e['training_rollouts_used_in_endpoint_rate'] is not False
                    or e['scope_successes_used_as_target_events'] is not False
                    or e['clock_is_total_search_compute'] is not False):
                raise ValueError('Trajectory event, censoring or endpoint roles disagree')
        s=selected[r];label=s['selected_name'];b=None
        if label=='start':endpoint_label='initial'
        elif label=='direct':endpoint_label='branch_endpoint';b=0
        elif label in ('candidate_1','candidate_2','candidate_3'):
            endpoint_label='branch_endpoint';b=int(label[-1])
        else:raise ValueError('Unknown selection action')
        point=indexed[r,endpoint_label,b,'target','full_pass']
        post=indexed[r,'selected_completion',None,'target','full_pass']
        if (s['selected_checkpoint']!=point['checkpoint'] or post['checkpoint']!=point['checkpoint']
                or s['completion_full_successes']!=post['observed_count'] or s['completion_draws']!=64
                or s['completion_used_for_selection'] is not False or s['challenger_updated'] is not False
                or s['round_loss_tokens_all_four_branches']!=4*524288):
            raise ValueError('Selected completion changed the original policy decision')
        previous=s['selected_checkpoint']
    if (data['observed_training_or_endpoint_event_branches']!=sum(e['event_observed'] for e in events.values())
            or data['right_censored_branches']!=sum(e['right_censored'] for e in events.values())):
        raise ValueError('Event totals disagree')
    expected_contrasts={(r,b,kind,m) for r in range(6) for b in (1,2,3)
                        for kind,metrics in [('target',names),('scope',scope)] for m in metrics}
    seen=set()
    for c in data['within_search_contrasts']:
        key=tuple(c[k] for k in ('round','candidate','kind','metric'))
        if key not in expected_contrasts or key in seen or c['reference']!=0:
            raise ValueError('Missing, duplicate or foreign paired contrast')
        seen.add(key);r,b,kind,m=key;lo,hi=c['paired_95_interval']
        raw=indexed[r,'branch_endpoint',b,kind,m]['observed_rate']-indexed[r,'branch_endpoint',0,kind,m]['observed_rate']
        identity=events[r,b]['optimizer_steps']==0 and events[r,0]['optimizer_steps']==0
        if (not math.isclose(c['difference'],raw,rel_tol=0,abs_tol=1e-12)
                or any(type(v) not in (int,float) or not math.isfinite(v) or not -1<=v<=1 for v in (lo,hi))
                or lo>hi or c['identical_policy_by_zero_update_provenance'] is not identity):
            raise ValueError('Raw draw contrast must remain separate from identity-adjusted credit')
    if seen!=expected_contrasts:raise ValueError('Incomplete paired comparison matrix')
    costs=data['generation_ledger']
    if len(costs)!=6 or {c['round'] for c in costs}!=set(range(6)):
        raise ValueError('Incomplete unique generation ledger')
    fields=('generated_initial_tokens','generated_training_tokens','generated_worker_endpoint_tokens','generated_selected_completion_tokens')
    for c in costs:
        if (c['loss_tokens']!=4*524288 or c['challenger_generated_tokens']!=0
                or any(type(c.get(k)) is not int or c[k]<0 for k in ('rollouts','binary_verifier_test_calls'))
                or any(type(c[k]) is not int or c[k]<0 for k in fields)
                or c['generated_training_tokens']<c['loss_tokens']
                or c['generated_tokens_all_phases']!=sum(c[k] for k in fields)
                or c['non_loss_generation_tokens']!=c['generated_tokens_all_phases']-c['loss_tokens']):
            raise ValueError('Search generation accounting does not reconcile')
    return indexed


def resource_totals(snapshot,registered):
    if not registered or snapshot.get('accounting_complete') is not True or snapshot.get('missing_individual_accounting_rows'):
        raise ValueError('All search allocations must have final accounting')
    rows=snapshot['allocations'];by_id={r['job_id']:r for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!=set(registered):raise ValueError('Foreign, missing or duplicate allocation')
    stages={stage:{'gpu_allocation_hours':0.,'cpu_allocation_core_hours':0.} for stage in ('prepare','worker','finish')}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    for identity,meta in registered.items():
        row=by_id[identity]
        if (any(row[k]!=v for k,v in meta.items()) or row['stage'] not in stages
                or row['state'].split()[0] not in terminal):raise ValueError('Wrong search ownership or live allocation')
        gpu,cpu,seconds=(row[k] for k in ('allocated_gpus','allocated_cpus','elapsed_seconds'))
        memory=row['allocated_cpu_memory_bytes']
        if (any(type(v) is not int or v<0 for v in (gpu,cpu,seconds)) or gpu>1 or cpu>8
                or (memory is not None and (not math.isfinite(memory) or not 0<=memory<=32*1024**3))):
            raise ValueError('Invalid or oversized search allocation')
        for k,expected in [('gpu_allocation_hours',gpu*seconds/3600),('cpu_allocation_core_hours',cpu*seconds/3600)]:
            if not math.isclose(row[k],expected,rel_tol=0,abs_tol=1e-10):raise ValueError('Allocation units do not reconcile')
            stages[row['stage']][k]+=expected
    total={k:sum(v[k] for v in stages.values()) for k in ('gpu_allocation_hours','cpu_allocation_core_hours')}
    if any(not math.isclose(snapshot[k],v,rel_tol=0,abs_tol=1e-10) for k,v in total.items()):
        raise ValueError('Final allocation totals disagree')
    return {'stages':stages,'totals':total,'historical_failed_allocations_included':True,
            'primary_control_followon_costs_kept_separately':True,'total_compute_matched':False}


def collect():
    if os.environ.get('VERGE_BOOK_SUITE')!='verge_book_v2' or os.environ.get('VERGE_SEARCH_SUITE')!='verge_search_v1':
        raise RuntimeError('Both explicit suite variables required')
    data=analysis.collect();validate_analysis(data)
    snapshot=resources.collect();registered,_=resources.registry(ctl.ROOT)
    hardware=resource_totals(snapshot,registered);states={x['job_id']:x['state'] for x in snapshot['allocations']}
    for r in range(6):
        jobs=ctl.read(ctl.ROOT/'jobs'/f'round_{r:02d}.json')
        current=[jobs['prepare'],jobs['finish']]+[f"{jobs['workers']}_{b}" for b in jobs.get('worker_indices',jobs['branches'])]
        if any(states.get(j)!='COMPLETED' for j in current):raise RuntimeError('A current search job did not finish successfully')
    return {'analysis':data,'resources':snapshot,'hardware':hardware,'synthetic':False,
            'report_rendering_complete':False,'search_block_complete':False,'suite_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    bundle=collect()
    if args.save:ctl.write(ctl.ROOT/'report_bundle.json',bundle)
    print('Search report values verified; rendering and artifacts remain incomplete; suite_complete=false')
