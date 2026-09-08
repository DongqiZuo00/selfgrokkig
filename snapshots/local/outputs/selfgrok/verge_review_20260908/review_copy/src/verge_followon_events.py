"""Target-only follow-on event accounting, separate from endpoint estimators."""
import copy

BUDGET = 262144


def first_training_event(state, items, masks, update):
    if not items or len(items) % 8 or len(items) != len(masks):
        raise ValueError('Complete target reward groups and loss masks are required')
    charged, generated = state['train_tokens'],state['generated_tokens']
    first = copy.deepcopy(state['first_target_success'])
    for start in range(0,len(items),8):
        group = items[start:start+8]
        allocated = masks[start:start+8]
        if (any(r.get('prompt_role') != 'target' or r['reward'] not in (0,1) for r in group)
                or len({r['instance_id'] for r in group}) != 1
                or [r['rollout_index'] for r in group] != list(range(8))):
            raise ValueError('A follow-on event must come from one binary target group')
        for row,mask in zip(group,allocated):
            if (len(mask) != row['completion_tokens'] or set(mask)-{0,1}
                    or not 0 < row['completion_tokens'] <= 2048):
                raise ValueError('Invalid group token accounting')
        charged += sum(sum(m) for m in allocated)
        generated += sum(r['completion_tokens'] for r in group)
        if charged > BUDGET:
            raise ValueError('Follow-on budget exceeded')
        if first is None and any(r['reward'] for r in group):
            first = {'role':'followon_target_training','update':update,
                'group_index':group[0]['reward_group'],
                'observation_loss_tokens':charged,'generated_tokens_through_group':generated,
                'clock':'End of first verified reward group containing a full success; not exact capability-acquisition time'}
    return first


def final_event(source, state, target_profile, completion_profile):
    if state['train_tokens'] != BUDGET:
        raise ValueError('Incomplete follow-ons cannot be labelled budget-censored completed runs')
    if target_profile['rollouts'] != 512 or completion_profile['rollouts'] != 64:
        raise ValueError('The fixed endpoint and completion evaluations are required')
    first = copy.deepcopy(state['first_target_success'])
    if (first is None) != (state['target_successes'] == 0):
        raise ValueError('Training success count and first-event record disagree')
    if first is None:
        for role,profile in (('followon_target_selection',target_profile),('followon_completion',completion_profile)):
            if profile['counts'][-1]:
                first = {'role':role,'observation_loss_tokens':BUDGET,
                         'clock':'Post-training observation at the completed loss-token budget'}
                break
    if first is not None and not 0 <= first['observation_loss_tokens'] <= BUDGET:
        raise ValueError('First-event time is outside the executed budget')
    return {'candidate_index':source['candidate_index'],'source_cohort':source['source_cohort'],
        'planned_loss_tokens':BUDGET,'executed_loss_tokens':BUDGET,'budget_complete':True,
        'event_observed':first is not None,
        'observation_loss_tokens':first['observation_loss_tokens'] if first else BUDGET,
        'first_event':first,
        'prior_source_event_recorded':bool(source['initial_target_full_successes'] or source['source_training_target_successes']),
        'target_training_full_successes':state['target_successes'],
        'target_endpoint_full_successes':target_profile['counts'][-1],
        'completion_full_successes':completion_profile['counts'][-1],
        'right_censored':first is None,
        'scope_successes_used_as_target_events':False,
        'training_rollouts_used_in_endpoint_rate':False}
