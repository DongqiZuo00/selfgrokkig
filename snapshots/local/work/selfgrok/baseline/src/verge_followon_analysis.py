"""All-candidate endpoint and censored-event analysis, without new model draws.

Rank comparisons stay inside each three-candidate source cohort. Aggregates are
descriptive sums over those comparisons, not pooled causal or iid-branch tests.
This intermediate JSON is not a completed report or experiment-book marker.
"""
import argparse
import json
import math
import verge_followon_controller as ctl
from verge_followon_protocol import ARMS,candidate_from_bank,validate_plan
from verge_followon_events import final_event
from verge_followon_ranks import summarize_cohort


def conditions():
    from verge_round_core import condition_names
    return condition_names()


def profile_rows(profile, *, kind, draws, context):
    if kind=='scope':
        names=('full_pass_rate','mean_case_fraction')
        values=[profile[k] for k in names];counts=[None,None]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Invalid scope endpoint rates')
    else:
        names=conditions();counts=profile['counts'];values=profile['rates']
        if (profile['rollouts']!=draws or profile['condition_names']!=names
                or len(counts)!=11 or len(values)!=11
                or any(type(n) is not int or not 0<=n<=draws for n in counts)
                or any(b>a for a,b in zip(counts,counts[1:]))
                or any(type(v) not in (int,float) or not math.isfinite(v)
                       or not math.isclose(v,n/draws,rel_tol=0,abs_tol=1e-12)
                       for n,v in zip(counts,values))):
            raise ValueError('Invalid fixed-checkpoint cumulative profile')
    return [dict(context,kind=kind,metric=k,rollouts=draws,observed_count=n,observed_rate=v)
            for k,n,v in zip(names,counts,values)]


def descriptive_aggregates(cohorts):
    """Preserve within-cohort comparisons, including null when not identifiable."""
    rows=[]
    for stratum in ('all_observations','pre_success','post_ignition'):
        for arm in ('all_arms',)+ARMS:
            subset=[c for c in cohorts if (arm=='all_arms' or c['source_arm']==arm)
                    and (stratum=='all_observations' or c['pre_success_analysis']==(stratum=='pre_success'))]
            for strategy in ('condition','uncertainty','random'):
                pairs=sum(c['strategies']['condition']['observed_order_concordance']['comparable_pairs'] for c in subset)
                credit=pairs*.5 if strategy=='random' else sum(
                    c['strategies'][strategy]['observed_order_concordance']['concordant_pair_credit'] for c in subset)
                for k in (1,2,3):
                    top=[c['strategies'][strategy]['top_k'][k-1] for c in subset]
                    fraction_sum=sum(t['mean_observed_event_fraction'] for t in top)
                    any_sum=sum(t['mean_any_observed_event_indicator'] for t in top)
                    rows.append({'stratum':stratum,'source_arm':arm,'strategy':strategy,'top_k':k,
                        'cohorts':len(subset),'candidates':3*len(subset),
                        'observed_events':sum(c['observed_events_within_budget'] for c in subset),
                        'right_censored':sum(c['right_censored_at_budget'] for c in subset),
                        'sum_cohort_top_k_event_fractions':fraction_sum,
                        'mean_cohort_top_k_event_fraction':fraction_sum/len(subset) if subset else None,
                        'sum_cohort_top_k_any_event_indicators':any_sum,
                        'mean_cohort_top_k_any_event_indicator':any_sum/len(subset) if subset else None,
                        'within_cohort_comparable_pairs':pairs,'concordant_pair_credit':credit,
                        'pair_weighted_descriptive_concordance':credit/pairs if pairs else None})
    return rows


def analyze(bank,strata,records):
    if len(records)!=72 or set(records)!=set(range(72)):
        raise ValueError('All 72 continuations are required; missing runs cannot become failures')
    if strata!=ctl.analysis_strata(bank):
        raise ValueError('Pre-success strata differ from the pre-follow-on source context')
    endpoint_rows=[];event_rows=[];cohorts=[];event_by_cohort={}
    for i in range(72):
        c=candidate_from_bank(bank,i);entry=records[i];initial=entry['initial'];result=entry['result']
        if (result['source_candidate']!=c or initial['checkpoint']!=c['source_checkpoint']
                or result.get('followon_segment_verified') is not True
                or result.get('followon_only') is not True or result.get('suite_complete') is not False
                or result['checkpoint']!=f"checkpoints/{ctl.version(i)}_direct/resume_u{result['training']['update']:04d}"):
            raise ValueError('Candidate identity or committed endpoint differs')
        event=final_event(c,result['training'],result['target'],result['completion'])
        if result['followon_event']!=event:
            raise ValueError('Reported event/censoring differs from fixed-budget observations')
        if (initial['target']['counts'][-1]!=c['initial_target_full_successes']
                or initial['training']['target_successes']!=c['source_training_target_successes']):
            raise ValueError('Saved source event provenance differs')
        stratum=strata['cohorts'][i//3]
        context={'array_index':i,'version':ctl.version(i),'source_cohort':c['source_cohort'],
            'source_arm':c['source_arm'],'source_round':c['source_round'],'candidate_index':c['candidate_index'],
            'source_solver_initial':c['source_solver_initial'],'pre_success_analysis':stratum['pre_success_analysis']}
        for label,summary,tokens in (('source',initial,0),('final',result,262144)):
            point=dict(context,endpoint=label,checkpoint=summary['checkpoint'],followon_loss_tokens=tokens,
                       source_endpoint_reused=label=='source')
            endpoint_rows+=profile_rows(summary['target'],kind='target',draws=512,context=point)
            endpoint_rows+=profile_rows(summary['scope'],kind='scope',draws=128,context=point)
        endpoint_rows+=profile_rows(result['completion'],kind='completion',draws=64,context=dict(
            context,endpoint='final_completion',checkpoint=result['checkpoint'],followon_loss_tokens=262144,
            source_endpoint_reused=False))
        event_by_cohort.setdefault(c['source_cohort'],[]).append(event)
        event_rows.append(dict(context,**{k:v for k,v in event.items() if k not in context},
            source_checkpoint=c['source_checkpoint'],final_checkpoint=result['checkpoint'],
            condition_rung_zero_based=c['condition_rung_zero_based'],condition_gain=c['condition_gain'],
            raw_endpoint_condition_gain=c['observed_endpoint_condition_gain'],uncertainty_score=c['uncertainty_score'],
            selected_in_primary=c['was_selected'],scope_safe_in_primary=c['scope_safe_observed'],
            eligible_in_primary=c['was_eligible'],scope_safe_vs_source_observed=result['scope_safe_vs_start'],
            optimizer_steps=result['training']['optimizer_steps']))
    for source,stratum in zip(bank['cohorts'],strata['cohorts']):
        item=summarize_cohort(source,event_by_cohort[source['cohort']])
        item.update(source_arm=stratum['source_arm'],source_round=stratum['source_round'],
                    pre_success_analysis=stratum['pre_success_analysis'])
        cohorts.append(item)
    return {'protocol':'verge_followon_v1_all_candidate_analysis','candidate_count':72,'cohort_count':24,
        'endpoint_rows':endpoint_rows,'trajectory_events':event_rows,'cohorts':cohorts,
        'descriptive_aggregates':descriptive_aggregates(cohorts),
        'endpoint_note':'Source and final are different fixed policies and different evaluation streams; no paired before/after confidence claim. Training is never pooled into either endpoint.',
        'event_note':'Grouped first observed success within the fixed budget, not exact capability acquisition. No event is right censoring, not proof of zero eventual success.',
        'aggregate_note':'Each numerator retains its within-cohort comparison; adaptive rounds and paired streams are dependent. No iid-branch, across-seed or causal aggregate confidence interval.',
        'source_endpoint_generation_charged_again':False,'total_compute_matched':False,
        'new_model_samples':0,'checkpoint_hash_scans':0,'official_test_opened':False,
        'followon_block_complete':False,'suite_complete':False}


def generation_cost(directory,result):
    from verge_control_analysis import rows
    state=result['training']
    ledger={'loss_tokens':state['train_tokens'],'nonzero_advantage_tokens':state['nonzero_advantage_tokens'],
        'generated_training_tokens':0,'generated_target_endpoint_tokens':0,'generated_scope_endpoint_tokens':0,
        'generated_completion_tokens':0,'generated_initial_endpoint_tokens':0,'raw_rollouts':0,
        'binary_verifier_test_calls':0,'source_endpoint_reused_without_generation':True}
    groups=[('generated_training_tokens',[directory/'train'/f'update_{i:04d}.jsonl' for i in range(1,state['update']+1)])]
    groups += [(f'generated_{kind}_endpoint_tokens',[directory/f'{kind}.jsonl']) for kind in ('target','scope')]
    groups += [('generated_completion_tokens',[directory/'completion.jsonl'])]
    training_rows=0
    for metric,paths in groups:
        for path in paths:
            for row in rows(path):
                tokens=row['completion_tokens'];cases=row['total_cases']
                if type(tokens) is not int or not 0<tokens<=2048 or type(cases) is not int or cases<1:
                    raise ValueError('Invalid generation accounting units')
                ledger[metric]+=tokens;ledger['raw_rollouts']+=1;ledger['binary_verifier_test_calls']+=cases
                training_rows+=int(metric=='generated_training_tokens')
    if ledger['generated_training_tokens']!=state['generated_tokens'] or training_rows!=state['rollouts']:
        raise ValueError('Committed generation ledger differs from training runtime')
    ledger['generated_completion_tokens_all_phases']=sum(ledger[k] for k,_ in groups)
    ledger['non_loss_generation_tokens']=ledger['generated_completion_tokens_all_phases']-ledger['loss_tokens']
    ledger['uncommitted_failed_attempt_generation_included']=False
    ledger['note']='Unique committed rollout files only, not duplicated response caches or source endpoint; Slurm allocation history separately includes failed/recovery jobs.'
    return ledger


def verify_endpoints(directory,result):
    """Reconcile all three saved endpoint profiles with native response records."""
    from verge_control_analysis import verified_endpoint,rows
    from verge_independent_sampling import PROTOCOL,validate_records,validate_response
    for kind,instances,draws in (('target',64,8),('scope',32,4)):
        verified_endpoint(directory,kind,result[kind],instances,draws)
    saved=ctl.read(directory/'completion.response.json');records=list(rows(directory/'completion.jsonl'))
    if (saved.get('backend_protocol')!=PROTOCOL or saved['request']['n']!=1
            or len(saved['request']['prompt'])!=64 or len(records)!=64):
        raise ValueError('Wrong final completion dimensions or independent backend')
    validate_response(saved['request'],saved['response']);validate_records(saved['request'],records)
    # The normal generator profile already contains the nested vectors. Check
    # their identities, binary entries, full verifier and counts without sampling.
    summary=result['completion'];keys=[f"{r['instance_id']}::{r['rollout_index']}" for r in records]
    if len(set(keys))!=64 or set(keys)!=set(summary['keyed_vectors']):
        raise ValueError('Completion profile identities differ from native records')
    vectors=[summary['keyed_vectors'][key] for key in keys]
    for row,vector in zip(records,vectors):
        if (len(vector)!=11 or any(type(v) is not int or v not in (0,1) for v in vector)
                or any(b>a for a,b in zip(vector,vector[1:])) or vector[-1]!=row['reward']
                or row['reward']!=int(row['passed_cases']==row['total_cases'])
                or type(row['total_cases']) is not int or not 0<=row['passed_cases']<=row['total_cases']
                or row['total_cases']<1 or row['max_tokens']!=2048
                or not 0<row['completion_tokens']==len(row['completion_token_ids'])<=2048):
            raise ValueError('Invalid completion native outcome or nested profile')
    if [sum(v[j] for v in vectors) for j in range(11)]!=summary['counts']:
        raise ValueError('Completion aggregate differs from per-response vectors')
    if ctl.read(directory/'completion_profile.json')!=summary:
        raise ValueError('Completion profile and final result differ')


def collect():
    marker=ctl.ROOT/'FOLLOWONS_FINISHED.json'
    if not marker.is_file() or ctl.read(marker).get('all_candidate_followons_finished') is not True:
        raise RuntimeError('Wait for all follow-on training before producing comparison output')
    plan=validate_plan(ctl.read(ctl.EXP/ctl.PLAN))
    bank=ctl.read(ctl.ROOT/'source_bank.json');strata=ctl.read(ctl.ROOT/'cohort_analysis_strata.json')
    if plan!=ctl.read(ctl.ROOT/'protocol_frozen.json') or bank!=ctl.read(ctl.ROOT/'source_bank_frozen.json'):
        raise RuntimeError('Source bank or analysis plan changed after submission')
    records={};costs=[]
    for i in range(72):
        result=ctl.completed(i,deep=True)
        if result is None:raise RuntimeError('Incomplete all-candidate follow-on block')
        directory=ctl.EXP/'raw_results'/ctl.version(i)
        frozen=ctl.read(directory/'round_frozen.json')
        source=result['source_candidate']
        original=ctl.EXP/'raw_results'/source['source_cohort']/'branches'/str(source['candidate_index'])/'complete.json'
        if ctl.read(original)!=frozen['initial']:
            raise RuntimeError('Reused source endpoint differs from its original completion record')
        stored=ctl.read(directory/'branches/0/endpoint.json')
        if any(stored[k]!=result[k] for k in ('checkpoint','target','scope')):
            raise RuntimeError('Mixed-checkpoint final endpoint')
        verify_endpoints(directory/'branches/0',result)
        records[i]={'initial':frozen['initial'],'result':result}
        costs.append(dict(array_index=i,version=ctl.version(i),**generation_cost(directory/'branches/0',result)))
    data=analyze(bank,strata,records)
    for c in data['cohorts']:c['analysis_plan_frozen']=True
    data.update(generation_ledger=costs,analysis_plan_frozen=True,synthetic=False)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    result=collect()
    if args.save:ctl.write(ctl.ROOT/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('endpoint_rows','trajectory_events','cohorts','descriptive_aggregates','generation_ledger')},indent=2))
