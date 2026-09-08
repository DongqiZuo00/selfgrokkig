"""Branch-local search events. Round completion and global compute are separate."""
import copy

BUDGET=524288


def first_training_event(state,items,masks,update):
    if not items or len(items)%8 or len(items)!=len(masks):raise ValueError('Complete binary groups required')
    charged,generated=state['train_tokens'],state['generated_tokens'];first=copy.deepcopy(state['first_target_success'])
    for start in range(0,len(items),8):
        group=items[start:start+8];allocated=masks[start:start+8]
        if (any(r.get('prompt_role')!='target' or r['reward'] not in (0,1) for r in group)
                or len({r['instance_id'] for r in group})!=1
                or [r['rollout_index'] for r in group]!=list(range(8))):raise ValueError('Invalid target reward group')
        for row,mask in zip(group,allocated):
            if len(mask)!=row['completion_tokens'] or set(mask)-{0,1} or not 0<row['completion_tokens']<=2048:
                raise ValueError('Invalid native group token accounting')
        charged+=sum(map(sum,allocated));generated+=sum(r['completion_tokens'] for r in group)
        if charged>BUDGET:raise ValueError('Search branch budget exceeded')
        if first is None and any(r['reward'] for r in group):
            first={'role':'search_target_training','update':update,'group_index':group[0]['reward_group'],
                'observation_loss_tokens':charged,'generated_tokens_through_group':generated,
                'clock':'Branch-local end of first verified group containing full success; not total search compute'}
    return first


def final_event(cfg,state,target):
    if state['train_tokens']!=BUDGET or target['rollouts']!=512:raise ValueError('Incomplete search branch is not censored completion')
    first=copy.deepcopy(state['first_target_success'])
    if (first is None)!=(state['target_successes']==0):raise ValueError('Training count/event disagree')
    if first is None and target['counts'][-1]:
        first={'role':'search_target_selection','observation_loss_tokens':BUDGET,
            'clock':'Post-training observation at branch-local completed loss budget'}
    if first and not 0<=first['observation_loss_tokens']<=BUDGET:raise ValueError('Event outside branch budget')
    return {'search_round':cfg['search_round'],'branch_index':cfg['search_branch_index'],
        'planned_loss_tokens':BUDGET,'executed_loss_tokens':BUDGET,'budget_complete':True,
        'event_observed':first is not None,'first_event':first,'right_censored':first is None,
        'observation_loss_tokens':first['observation_loss_tokens'] if first else BUDGET,
        'target_training_full_successes':state['target_successes'],'target_endpoint_full_successes':target['counts'][-1],
        'training_rollouts_used_in_endpoint_rate':False,'scope_successes_used_as_target_events':False,
        'selected_round_completion_pending':True,'clock_is_total_search_compute':False}
