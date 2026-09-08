"""Collect every primary candidate for later equal-budget follow-on training.

No model is loaded, no new rollout is sampled and no checkpoint is hashed.
This is a source manifest, not a follow-on run or a choice of ranking policy.
"""
import argparse
import copy
import math
import re
import verge_book_controller as book


def target_profile(endpoint):
    p = endpoint['target']
    if (p['rollouts'] != 512 or len(p['counts']) != 11 or len(p['rates']) != 11
            or any(type(n) is not int or not 0 <= n <= 512 for n in p['counts'])
            or any(b > a for a,b in zip(p['counts'],p['counts'][1:]))
            or any(not math.isclose(rate,count/512,rel_tol=0,abs_tol=1e-12) for rate,count in zip(p['rates'],p['counts']))):
        raise RuntimeError('Invalid fixed-checkpoint target profile')
    return p


def uncertainty_from_training(branch, proposal):
    stats = branch['training']['curriculum_stage_reward_counts']
    if not stats or len(stats) != len(proposal['spec']['stages']):
        raise RuntimeError('Missing curriculum-only stage success counts')
    for s in stats:
        if (type(s['draws']) is not int or type(s['successes']) is not int
                or not 0 <= s['successes'] <= s['draws'] or s['draws'] < 1):
            raise RuntimeError('Uncertainty is unavailable, not a default zero')
    return sum(1-2*abs(s['successes']/s['draws']-.5) for s in stats)/len(stats)


def cohort_sources(frozen, decision, branches):
    cfg = frozen['config']
    if (cfg.get('book_suite') != 'verge_book_v2' or cfg.get('independent_prompt_streams') is not True
            or cfg.get('sampling_protocol') != 'per_prompt_disjoint_seed_ranges_v1'
            or cfg.get('seed') != 42 or cfg.get('completion_tokens') != 2048
            or cfg.get('train_tokens_per_branch') != 524288):
        raise RuntimeError('Only the corrected primary contract is a follow-on source')
    arm,r,name = cfg['book_arm'],cfg['book_round'],cfg['protocol_version']
    if name != f'verge_book_v2_{arm}_r{r:02d}' or arm not in book.PRIMARY_ARMS or not 0 <= r < 6:
        raise RuntimeError('Foreign primary source namespace')
    proposals = {p['index']:p for p in frozen['generation']['candidates']}
    if (set(proposals) != {1,2,3} or len(frozen['generation']['candidates']) != 3
            or not all(p['valid'] for p in proposals.values()) or set(branches) != {0,1,2,3}):
        raise RuntimeError('A repaired primary cohort must contain all three rendered candidates and direct')
    if len(decision['target_condition_gains']) != 3:
        raise RuntimeError('Missing frozen condition credit')
    evaluated = {c['index']:c for c in decision['evaluated_candidates']}
    if set(evaluated) != {1,2,3} or len(decision['evaluated_candidates']) != 3:
        raise RuntimeError('Missing candidate decision context')
    rung = decision['reward_rung_zero_based']
    if type(rung) is not int or not 0 <= rung < 11 or decision.get('endpoint_only_credit') is not True:
        raise RuntimeError('Condition credit must refer to a fixed-endpoint rung')
    direct = branches[0]
    direct_profile = target_profile(direct)
    candidates = []
    for i,branch in branches.items():
        state = branch['training']
        if (state['train_tokens'] != 524288 or type(state['update']) is not int
                or not 0 < state['update'] <= 100 or type(state['optimizer_steps']) is not int
                or not 0 <= state['optimizer_steps'] <= state['update']):
            raise RuntimeError('A source branch has not finished the primary budget')
        stem = f'checkpoints/{name}_{"direct" if i == 0 else "candidate_"+str(i)}/resume_u'
        if not re.fullmatch(re.escape(stem)+r'\d{4}',branch['checkpoint']) or int(branch['checkpoint'][-4:]) != state['update']:
            raise RuntimeError('Foreign or stale source branch checkpoint')
        profile = target_profile(branch)
        if i == 0:
            continue
        observed = profile['rates'][rung]-direct_profile['rates'][rung]
        identical = state['optimizer_steps'] == direct['training']['optimizer_steps'] == 0
        credit = 0. if identical else observed
        if not math.isclose(credit,decision['target_condition_gains'][i-1],rel_tol=0,abs_tol=1e-12):
            raise RuntimeError('Saved condition credit differs from its endpoint/provenance definition')
        candidates.append({
            'source_cohort':name,'source_arm':arm,'source_round':r,'candidate_index':i,
            'source_checkpoint':branch['checkpoint'],'source_solver_initial':cfg['solver_start'],
            'curriculum_spec':copy.deepcopy(proposals[i]['spec']),
            'condition_rung_zero_based':rung,'condition_gain':credit,
            'observed_endpoint_condition_gain':observed,'zero_update_policy_identity':identical,
            'uncertainty_score':uncertainty_from_training(branch,proposals[i]),
            'uncertainty_source':'Curriculum-only training stage success counts; no new stage probes',
            'was_selected':decision['selected']['name'] == f'candidate_{i}',
            'scope_safe_observed':evaluated[i]['scope_safe'],'was_eligible':evaluated[i]['eligible'],
            'initial_target_full_successes':profile['counts'][-1],'initial_target_endpoint_draws':512,
            'source_optimizer_steps':state['optimizer_steps'],
            'source_training_target_successes':state['target_successes'],
            'source_first_training_target_success':copy.deepcopy(state['first_target_success']),
            'follow_on_loss_tokens':262144,
            'endpoint_source':f'raw_results/{name}/branches/{i}/target.jsonl',
        })
    return {'cohort':name,'source_solver_initial':cfg['solver_start'],
            'direct_reference_checkpoint':direct['checkpoint'],
            'direct_reference_endpoint':f'raw_results/{name}/branches/0/target.jsonl',
            'candidates':sorted(candidates,key=lambda c:c['candidate_index'])}


def collect():
    if book.SUITE != 'verge_book_v2' or book.os.environ.get('VERGE_BOOK_SUITE') != 'verge_book_v2':
        raise RuntimeError('Explicit corrected suite is required')
    marker = book.EXP/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json'
    if not marker.is_file() or book.read(marker).get('primary_block_complete') is not True or book.read(marker).get('outer_rounds') != 24:
        raise RuntimeError('Wait for all 24 primary cohorts; do not construct a winner-only partial bank')
    plan = book.read(book.EXP/'raw_results/verge_book_v2/protocol_frozen.json')
    if plan['follow_on_tokens_per_candidate'] != 262144:
        raise RuntimeError('Unexpected follow-on budget')
    cohorts = []
    for arm,r in book.schedule():
        book.verified_round(arm,r)
        name = f'verge_book_v2_{arm}_r{r:02d}'
        root = book.EXP/'raw_results'/name
        frozen = book.read(root/'round_frozen.json')
        branches = {i:book.read(root/'branches'/str(i)/'complete.json') for i in range(4)}
        cohort = cohort_sources(frozen,book.read(root/'decision.json'),branches)
        if cohort['cohort'] != name:
            raise RuntimeError('Scheduled cohort and saved source disagree')
        initial_full = target_profile(frozen['initial'])['counts'][-1]
        completion = book.read(root/'completion_profile.json')
        if (completion['rollouts'] != 64 or len(completion['counts']) != 11
                or any(type(n) is not int or not 0 <= n <= 64 for n in completion['counts'])):
            raise RuntimeError('Missing training-side post-selection event context')
        cohort['source_event_context'] = {
            'initial_endpoint_full_successes':initial_full,
            'branch_endpoint_full_successes':[branches[i]['target']['counts'][-1] for i in range(4)],
            'branch_training_full_successes':[branches[i]['training']['target_successes'] for i in range(4)],
            'post_selection_full_successes':completion['counts'][-1],
        }
        context = cohort['source_event_context']
        cohort['no_full_event_recorded_in_source_cohort'] = not any(
            [initial_full,context['post_selection_full_successes']]
            +context['branch_endpoint_full_successes']+context['branch_training_full_successes'])
        for checkpoint in [cohort['direct_reference_checkpoint']]+[c['source_checkpoint'] for c in cohort['candidates']]:
            path = (book.EXP/checkpoint).resolve()
            if not path.is_relative_to((book.EXP/'checkpoints').resolve()):
                raise RuntimeError('Checkpoint escaped the experiment')
            for filename in ('verge_committed.json','adapter_model.safetensors','state.pt'):
                if not (path/filename).is_file() or (path/filename).stat().st_size == 0:
                    raise RuntimeError('A required candidate checkpoint was not retained')
            if book.read(path/'verge_committed.json')['update'] != int(checkpoint[-4:]):
                raise RuntimeError('Source checkpoint commit and update disagree')
        cohorts.append(cohort)
    if len(cohorts) != 24 or sum(len(c['candidates']) for c in cohorts) != 72:
        raise RuntimeError('Incomplete all-candidate follow-on coverage')
    return {'protocol':'verge_followon_v1_source_bank','cohorts':cohorts,'candidate_count':72,
            'reference_metadata_count':24,'follow_on_candidate_loss_tokens':72*262144,
            'eligibility_filter_applied':False,'selected_candidate_filter_applied':False,
            'checkpoint_identity_deduplication':False,
            'comparison_strata':'Preserve source cohort/arm/round/initial Solver and rung; no pooled cross-cohort causal ranking',
            'pre_success_note':'Retain all candidates; source-cohort event context is metadata for a separately frozen pre-success analysis, not a post-follow-on outcome filter',
            'direct_reference_training_authorized_by_this_bank':False,
            'follow_on_protocol_frozen':False,'follow_on_training_submitted':False,
            'ranking_policy_frozen':False,'suite_complete':False,'official_test_opened':False,
            'new_model_samples':0,'checkpoint_hash_scans':0,
            'pending':'Separate follow-on optimizer/runner, ranking tie rules, random ranking and censoring protocol must be frozen before training'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true')
    args = parser.parse_args()
    result = collect()
    if args.save:
        destination = book.EXP/'raw_results/verge_followon_v1/source_bank.json'
        if destination.exists() and book.read(destination) != result:
            raise RuntimeError('Refusing to replace a changed all-candidate source bank')
        if not destination.exists():
            book.write(destination,result)
    print(book.json.dumps({k:v for k,v in result.items() if k != 'cohorts'},indent=2))
