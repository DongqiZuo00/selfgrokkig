"""Fixed-endpoint search summaries; no new samples or outcome-dependent choices."""
import argparse
import copy
import json
import verge_search_controller as ctl
from verge_search_protocol import validate_plan,validate_round_config,worker_config,select_round
from verge_search_events import final_event
from verge_followon_analysis import profile_rows
from verge_control_analysis import rows,verified_endpoint,paired_contrast


def analyze(entries):
    if set(entries)!=set(range(6)):raise ValueError('All six completed search rounds are required')
    profiles=[];events=[];selections=[];previous=None
    for r in range(6):
        entry=entries[r];frozen=entry['frozen'];cfg=frozen['config'];validate_round_config(cfg)
        complete=entry['complete'];branches=entry['branches'];completion=entry['completion']
        if (cfg['search_round']!=r or cfg['protocol_version']!=ctl.version(r)
                or complete.get('search_round_complete') is not True
                or complete.get('search_block_complete') is not False or complete.get('suite_complete') is not False
                or complete.get('completed_branches')!=4 or complete.get('round_loss_tokens')!=4*524288
                or complete.get('official_test_opened') is not False):raise ValueError('Incomplete or foreign search round')
        if previous is not None and cfg['solver_start']!=previous:raise ValueError('Search lineage changed after selection')
        if frozen['initial']['checkpoint']!=cfg['solver_start']:raise ValueError('Mixed-checkpoint initial profile')
        expected=select_round(frozen,branches)
        if entry['decision']!=expected or complete['selected']!=expected['selected']:
            raise ValueError('Search decision differs from its frozen endpoint selector')
        context={'round':r,'round_version':ctl.version(r),'endpoint':'initial','branch_index':None,
                 'checkpoint':cfg['solver_start'],'branch_training_loss_tokens':0}
        profiles+=profile_rows(frozen['initial']['target'],kind='target',draws=512,context=context)
        profiles+=profile_rows(frozen['initial']['scope'],kind='scope',draws=128,context=context)
        for b in range(4):
            result=branches[b];state=result['training']
            if result.get('search_segment_verified') is not True or result.get('suite_complete') is not False:
                raise ValueError('Unverified search worker is not a completed observation')
            event=final_event(worker_config(cfg,b),state,result['target'])
            if result['search_event']!=event:raise ValueError('Event record differs from the original fixed-budget trajectory')
            point=dict(context,endpoint='branch_endpoint',branch_index=b,checkpoint=result['checkpoint'],branch_training_loss_tokens=524288)
            profiles+=profile_rows(result['target'],kind='target',draws=512,context=point)
            profiles+=profile_rows(result['scope'],kind='scope',draws=128,context=point)
            events.append(dict(round=r,version=ctl.version(r,b),checkpoint=result['checkpoint'],
                updates=state['update'],optimizer_steps=state['optimizer_steps'],**copy.deepcopy(event)))
        if (completion.get('checkpoint')!=expected['selected']['checkpoint']
                or complete.get('completion_draws')!=64 or complete.get('completion_full_successes')!=completion['counts'][-1]
                or complete.get('completion_used_for_selection') is not False):raise ValueError('Selected completion belongs to another decision')
        point=dict(context,endpoint='selected_completion',checkpoint=completion['checkpoint'],branch_training_loss_tokens=None)
        profiles+=profile_rows(completion,kind='target',draws=64,context=point)
        selected=expected['selected'];previous=selected['checkpoint']
        selections.append({'round':r,'selected_name':selected['name'],'selected_checkpoint':previous,
            'selection_condition':expected['selection_rung_zero_based'],'selected_gain':selected['gain'],
            'eligible_branches':[x['index'] for x in expected['evaluated_candidates'] if x['eligible']],
            'selection_probabilities':expected['selection_probabilities'],
            'round_loss_tokens_all_four_branches':4*524288,'completion_full_successes':completion['counts'][-1],
            'completion_draws':64,'completion_used_for_selection':False,'challenger_updated':False})
    return {'protocol':'verge_search_v1_endpoint_analysis','rounds':6,'branches':24,'endpoint_rows':profiles,
        'trajectory_events':events,'round_selections':selections,'total_loss_tokens':12582912,
        'observed_training_or_endpoint_event_branches':sum(e['event_observed'] for e in events),
        'right_censored_branches':sum(e['right_censored'] for e in events),
        'search_block_complete':False,'suite_complete':False,'official_test_opened':False,'new_model_samples':0,
        'total_compute_matched':False,'cross_primary_paired_comparisons':False,
        'event_note':'A branch-local first-success coordinate is not total search compute or global wall time. Selected completion is separate; trajectory draws never estimate fixed-checkpoint rates.',
        'statistical_note':'Within-round shared-stream contrasts only; exploratory two-level paired intervals, no multiplicity correction or cross-training-seed inference. Raw draw differences for zero-update identical policies are not learning gains; selector identity-adjusted gains are retained separately. Zero-width bootstrap intervals at zero observed success are not population upper bounds.'}


def generation_cost(round_directory,entry):
    result={'round':entry['frozen']['config']['search_round'],'loss_tokens':4*524288,
        'generated_initial_tokens':0,'generated_training_tokens':0,'generated_worker_endpoint_tokens':0,
        'generated_selected_completion_tokens':0,'rollouts':0,'binary_verifier_test_calls':0,'challenger_generated_tokens':0}
    groups=[('generated_initial_tokens',[round_directory/'initial'/(k+'.jsonl') for k in ('target','scope')]),
            ('generated_selected_completion_tokens',[round_directory/'completion.jsonl'])]
    expected_training=0
    for b,worker in entry['branches'].items():
        directory=ctl.EXP/'raw_results'/ctl.version(result['round'],b)/'branches/0';state=worker['training']
        expected_training+=state['generated_tokens']
        groups += [('generated_training_tokens',[directory/'train'/f'update_{i:04d}.jsonl' for i in range(1,state['update']+1)]),
                   ('generated_worker_endpoint_tokens',[directory/(k+'.jsonl') for k in ('target','scope')])]
    for label,paths in groups:
        for path in paths:
            for row in rows(path):
                result[label]+=row['completion_tokens'];result['rollouts']+=1
                result['binary_verifier_test_calls']+=row['total_cases']
    if result['generated_training_tokens']!=expected_training or expected_training<result['loss_tokens']:
        raise ValueError('Unique training-token ledger differs from committed search workers')
    result['generated_tokens_all_phases']=sum(result[k] for k in ('generated_initial_tokens','generated_training_tokens',
        'generated_worker_endpoint_tokens','generated_selected_completion_tokens'))
    result['non_loss_generation_tokens']=result['generated_tokens_all_phases']-result['loss_tokens']
    return result


def collect():
    plan=validate_plan(ctl.read(ctl.ROOT/'protocol_frozen.json'))
    if plan!=ctl.read(ctl.EXP/ctl.PLAN):raise RuntimeError('Search plan changed after submission')
    entries={};contrasts=[];costs=[]
    for r in range(6):
        complete=ctl.completed_round(r,deep=True)
        if complete is None:raise RuntimeError('All six verified search rounds must finish before analysis')
        directory=ctl.ROOT/'rounds'/ctl.version(r);frozen=ctl.read(directory/'round_frozen.json')
        branches={b:ctl.completed_worker(r,b,deep=True) for b in range(4)}
        entry={'complete':complete,'frozen':frozen,'branches':branches,'decision':ctl.read(directory/'decision.json'),
               'completion':ctl.read(directory/'completion_profile.json')};entries[r]=entry
        endpoints={}
        for b,worker in branches.items():
            output=ctl.EXP/'raw_results'/ctl.version(r,b)/'branches/0'
            stored=ctl.read(output/'endpoint.json')
            if any(stored[k]!=worker[k] for k in ('checkpoint','target','scope')):raise RuntimeError('Worker endpoint pointer changed')
            for kind,n,draws in (('target',64,8),('scope',32,4)):
                endpoints[b,kind]=verified_endpoint(output,kind,worker[kind],n,draws)
        for b in (1,2,3):
            for kind in ('target','scope'):
                for stat in paired_contrast(endpoints[b,kind],endpoints[0,kind],plan['bootstrap_replicates']):
                    contrasts.append(dict(round=r,candidate=b,reference=0,kind=kind,
                        identical_policy_by_zero_update_provenance=branches[b]['training']['optimizer_steps']==0
                            and branches[0]['training']['optimizer_steps']==0,**stat))
        costs.append(generation_cost(directory,entry))
    result=analyze(entries);result.update(within_search_contrasts=contrasts,generation_ledger=costs,
        bootstrap_replicates=plan['bootstrap_replicates'],
        cost_note='Unique committed rollout files only, not cache copies. Shared initial counted once and selected completion once per round. Hardware/recovery allocations are separate; these token counts are not FLOPs. Condition reexecution is excluded from verifier counts.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    result=collect()
    if args.save:ctl.write(ctl.ROOT/'endpoint_analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('endpoint_rows','trajectory_events','round_selections','within_search_contrasts','generation_ledger')},indent=2))
