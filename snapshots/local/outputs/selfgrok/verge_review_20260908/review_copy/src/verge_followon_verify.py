"""Validate follow-on training receipts and endpoint/event separation."""
import re
from common import read_json,read_jsonl,group_advantages
from verge_followon_protocol import PATTERN,validate_prepared
from verge_followon_events import first_training_event,final_event,BUDGET
from verge_independent_sampling import validate_records,validate_response,PROTOCOL


def require(condition, message):
    if not condition:raise ValueError(message)


def validate_segment(frozen, result, directory):
    cfg=frozen['config'];validate_prepared(cfg,frozen)
    require(re.fullmatch(PATTERN,cfg['protocol_version']) is not None and cfg.get('followon_only') is True,
            'Not an isolated follow-on configuration')
    state=result['training']
    require(type(state['update']) is int and 0 < state['update'] <= 100,'Invalid update count')
    require(cfg['train_tokens_per_branch']==BUDGET and cfg['followon_phases']==[
        {'kind':'target','reward_mode':'binary','tokens':BUDGET}], 'Changed target-only budget')
    require(result['checkpoint']==f"checkpoints/{cfg['protocol_version']}_direct/resume_u{state['update']:04d}",
            'Endpoint is not this continuation final checkpoint')
    require(result.get('fresh_optimizer') is True and result.get('weight_decay')==0.
            and result.get('kl_reference')==cfg['solver_start'] and result.get('binary_rewards_only') is True,
            'Continuation optimizer/reward contract changed')
    totals={'train_tokens':0,'generated_tokens':0,'first_target_success':None,
            'target_successes':0,'nonzero_advantage_tokens':0,'optimizer_steps':0,'rollouts':0}
    masked_groups=0;expected_group=0
    for update in range(1,state['update']+1):
        rows=read_jsonl(directory/'train'/f'update_{update:04d}.jsonl')
        allocation=read_json(directory/'train'/f'update_{update:04d}_allocation.json')
        summary=read_json(directory/'train'/f'update_{update:04d}_summary.json')
        require(allocation['stage']==0 and allocation['policy_checkpoint_update']==update-1
                and allocation['advantage_source']=='binary_complete_target_reward', 'Wrong policy/reward source')
        require(rows and len(rows)%8==0,'Incomplete target reward group')
        rewards=[r['reward'] for r in rows]
        require(set(rewards)<={0,1},'Partial rewards entered Solver training')
        advantages=group_advantages(rewards,8)
        require(allocation['advantages']==advantages,'Advantage differs from binary group reward')
        masks=allocation['loss_masks']
        require(len(masks)==len(rows),'Missing loss masks')
        for start in range(0,len(rows),8):
            group=rows[start:start+8]
            request={'prompt':[group[0]['prompt_token_ids']],'n':8,'seed':group[0]['sampling_parent_seed']}
            validate_records(request,group)
            require(all(r['reward_group']==expected_group for r in group),'Reused or skipped training group')
            expected_group+=1
            charged=sum(sum(m) for m in masks[start:start+8])
            generated=sum(r['completion_tokens'] for r in group)
            if charged < generated:
                masked_groups+=1
                require(update==state['update'] and start+8==len(rows),'Masking outside final budget boundary')
        for row,mask in zip(rows,masks):
            require(row['sampling_protocol']==PROTOCOL and row['max_tokens']==2048
                    and len(row['completion_token_ids'])==row['completion_tokens']==len(mask),
                    'Invalid native action accounting')
        totals['first_target_success']=first_training_event(totals,rows,masks,update)
        used=sum(sum(m) for m in masks)
        require(used>0,'No charged token progress')
        active=sum(sum(m)*int(abs(a)>1e-12) for m,a in zip(masks,advantages))
        should_step=active>0 or totals['optimizer_steps']>0
        require(summary['optimizer_step'] is should_step,'Unexpected skipped or spurious optimizer step')
        totals['optimizer_steps']+=int(should_step)
        totals['train_tokens']+=used
        totals['generated_tokens']+=sum(r['completion_tokens'] for r in rows)
        totals['nonzero_advantage_tokens']+=active
        totals['target_successes']+=sum(rewards)
        totals['rollouts']+=len(rows)
    require(masked_groups<=1,'Repeated budget masks')
    for key,value in totals.items():require(state[key]==value,f'Runtime total differs: {key}')
    require(state['train_tokens']==BUDGET and state['used_by_stage']==[BUDGET]
            and state['group_by_stage']==[expected_group] and state['target_loss_tokens']==BUDGET,
            'Unmatched follow-on loss budget')
    for kind,count in (('target',512),('scope',128),('completion',64)):
        path=directory/f'{kind}.jsonl'
        saved=read_json(path.with_suffix('.response.json'));records=read_jsonl(path)
        require(saved['backend_protocol']==PROTOCOL and len(records)==count,'Missing fixed endpoint records')
        validate_response(saved['request'],saved['response']);validate_records(saved['request'],records)
        if kind!='scope':
            profile=result['target' if kind=='target' else 'completion']
            require(profile['rollouts']==count and len(profile['counts'])==11
                    and all(type(n) is int and 0<=n<=count for n in profile['counts'])
                    and all(b<=a for a,b in zip(profile['counts'],profile['counts'][1:])),
                    'Invalid cumulative endpoint profile')
            require(profile['counts'][-1]==sum(r['reward'] for r in records),
                    'Reported full count differs from raw verifier outcomes')
    expected=final_event(cfg['followon_source'],state,result['target'],result['completion'])
    require(result['followon_event']==expected,'Event or censoring record differs from original receipts')
    return {'followon_segment_verified':True,'followon_comparison_complete':False,'suite_complete':False}
