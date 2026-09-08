"""Bounded target-only search configuration and the unchanged endpoint selector.

No runtime route, model draws or Slurm submission is enabled by this module.
The controller must freeze the whole round before any of its four workers run.
"""
import copy
import math
import re
from pathlib import Path
from verge_repair_config import repair_config

SUITE='verge_search_v1'
ROOT_CHECKPOINT='checkpoints/verge_book_v2_initial_solver/resume_u0000'
PLAN='manifests/verge_search_v1_target_only_protocol.json'


def validate_plan(plan):
    required={'protocol':'verge_search_v1_target_only_branch_search','source_suite':'verge_book_v2',
        'backbone':'mistralai/Ministral-3-3B-Instruct-2512-BF16','training_seed':42,
        'initial_solver':ROOT_CHECKPOINT,'outer_rounds':6,'branches_per_round':4,'reference_branch':0,
        'search_branches':[1,2,3],'loss_tokens_per_branch':524288,'total_loss_tokens':12582912,
        'target_reward':'binary_complete_verifier','target_prompt_fraction':1.,'curriculum_proposals':0,
        'challenger_updates':0,'maximum_updates_per_branch':100,'completion_cap':2048,'group_size':8,
        'update_threshold_tokens':16384,'solver_learning_rate':1e-5,'weight_decay':0.,'kl_beta':.02,
        'sampling_protocol':'per_prompt_disjoint_seed_ranges_v1','target_selection_instances':64,
        'target_selection_samples':8,'scope_instances':32,'scope_samples':4,'completion_instances':64,
        'completion_samples':1,'reward_switch_q':4,'saturation_margin_delta':.1,'selector_beta':2,
        'bootstrap_replicates':2000,'scope_max_drop':0.,'no_early_success_stop':True,'all_branches_retained':True,
        'worker_gpus':1,'worker_cpus':8,'worker_cpu_memory_gib':32,'worker_wall_limit_hours':2,
        'maximum_concurrent_workers':2,'maximum_total_cpu_memory_gib':64,
        'schedule_after':'verge_followon_v1 verified block completion','official_test_opened':False,
        'extra_diagnostic_samples':0,'checkpoint_hash_scans':0,'total_compute_matched':False}
    for key,value in required.items():
        if plan.get(key)!=value or type(plan.get(key)) is not type(value):
            raise ValueError(f'Changed target-only search setting: {key}')
    for key in ('optimizer_state','kl_reference','branch_training_randomness','endpoint_randomness',
                'selector','zero_update_identity','budget_note','single_seed_note'):
        if not isinstance(plan.get(key),str) or not plan[key].strip():raise ValueError('Missing search interpretation contract')
    return plan


def version(round_index,branch=None):
    if type(round_index) is not int or not 0<=round_index<6:raise ValueError('Search round outside six-round limit')
    name=f'{SUITE}_r{round_index:02d}'
    if branch is None:return name
    if type(branch) is not int or not 0<=branch<4:raise ValueError('Search branch outside four-branch limit')
    return f'{name}_b{branch}'


def valid_source(checkpoint,round_index):
    if checkpoint==ROOT_CHECKPOINT:return True
    match=re.fullmatch(r'checkpoints/verge_search_v1_r(\d\d)_b([0-3])_direct/resume_u(\d{4})',str(checkpoint))
    return bool(match and 0<=int(match[1])<round_index and 0<=int(match[3])<=100)


def round_config(exp,round_index,plan,previous=None):
    validate_plan(plan);name=version(round_index)
    start=ROOT_CHECKPOINT
    if round_index:
        if (not previous or previous.get('protocol_version')!=version(round_index-1)
                or previous.get('search_round_complete') is not True or previous.get('suite_complete') is not False):
            raise ValueError('Need the immediately preceding completed search-round selection')
        start=previous['selected']['checkpoint']
        if not valid_source(start,round_index):raise ValueError('Foreign or future search source checkpoint')
    elif previous is not None:raise ValueError('First search round cannot inherit an existing learned trajectory')
    cfg=repair_config('verge_mistral_repair_v2',Path(exp))
    cfg.update(protocol_version=name,target_only_search=True,search_suite=SUITE,search_round=round_index,
        search_protocol=PLAN,solver_start=start,solver_rank=16,solver_learning_rate=1e-5,
        outer_rounds=6,alpha=1.,eta=1.,weight_decay=0.,kl_beta=.02,
        train_tokens_per_branch=524288,maximum_updates_per_branch=100,
        independent_prompt_streams=True,sampling_protocol='per_prompt_disjoint_seed_ranges_v1',
        random_stream_id=f'{SUITE}_paired_r{round_index:02d}',proposal_count=0,stage_probe_instances=0,
        challenger_start=None,teacher_update_enabled=False,final_evaluation_allowed=False,stop_after_first_success=False,
        source_primary_suite='verge_book_v2',search_branches=[0,1,2,3],
        budget_note=plan['budget_note'],scope='One four-branch target-only search round; not the full experiment book',
        explicit_pilot_differences=[plan['single_seed_note'],plan['branch_training_randomness'],
            'All temporary branches train only on the unchanged complete target verifier',
            'No Challenger proposal generation or update; same target-conditioned endpoint selector',
            'No new eligibility audit, checkpoint hash scan or official-test access'])
    for key in ('book_suite','book_atomic_round','book_arm','control_only','followon_only'):cfg.pop(key,None)
    return validate_round_config(cfg)


def validate_round_config(cfg):
    r=cfg.get('search_round');name=version(r)
    fixed={'protocol_version':name,'target_only_search':True,'search_suite':SUITE,'search_protocol':PLAN,
        'backbone':'mistralai/Ministral-3-3B-Instruct-2512-BF16','seed':42,'solver_rank':16,
        'solver_learning_rate':1e-5,'weight_decay':0.,'kl_beta':.02,'outer_rounds':6,
        'train_tokens_per_branch':524288,'maximum_updates_per_branch':100,'completion_tokens':2048,
        'token_update_threshold':16384,'alpha':1.,'eta':1.,'proposal_count':0,'stage_probe_instances':0,
        'independent_prompt_streams':True,'sampling_protocol':'per_prompt_disjoint_seed_ranges_v1',
        'random_stream_id':f'{SUITE}_paired_r{r:02d}','selection_samples_per_instance':8,'endpoint_instances':64,
        'scope_instances':32,'scope_samples':4,'completion_instances':64,'reward_switch_q':4,
        'selector_beta':2.,'bootstrap_replicates':2000,'scope_max_drop':0.,'repair_integrated':True,
        'final_evaluation_allowed':False,'stop_after_first_success':False,'teacher_update_enabled':False,
        'solver_top_p':1.,'solver_top_k':-1,'solver_temperature':1.}
    if (any(cfg.get(k)!=v for k,v in fixed.items()) or not valid_source(cfg.get('solver_start'),r)
            or any(cfg.get(k) for k in ('book_suite','book_atomic_round','control_only','followon_only'))):
        raise ValueError('Changed or foreign target-only search configuration')
    return cfg


def worker_config(round_cfg,branch):
    validate_round_config(round_cfg)
    r=round_cfg['search_round'];name=version(r,branch)
    if (round_cfg['protocol_version']!=version(r) or round_cfg.get('target_only_search') is not True
            or round_cfg.get('search_suite')!=SUITE):raise ValueError('Not an isolated search round')
    cfg=copy.deepcopy(round_cfg)
    cfg.update(protocol_version=name,search_round_version=version(r),search_branch_index=branch,
        training_random_stream_id=f'{SUITE}_train_r{r:02d}_b{branch}',
        search_phases=[{'kind':'target','reward_mode':'binary','tokens':524288}])
    return cfg


def validate_profile(endpoint):
    p=endpoint['target'];n=p['rollouts'];counts=p['counts'];rates=p['rates']
    if (n!=512 or len(counts)!=11 or len(rates)!=11
            or any(type(c) is not int or not 0<=c<=512 for c in counts)
            or any(b>a for a,b in zip(counts,counts[1:]))
            or any(not math.isclose(rate,c/512,rel_tol=0,abs_tol=1e-12) for rate,c in zip(rates,counts))):
        raise ValueError('Invalid fixed-checkpoint search profile')
    if any(not math.isfinite(endpoint['scope'][k]) or not 0<=endpoint['scope'][k]<=1
            for k in ('full_pass_rate','mean_case_fraction')):raise ValueError('Invalid search scope profile')


def prepared_round(cfg,initial,selection_draw):
    validate_round_config(cfg)
    if initial['checkpoint']!=cfg['solver_start']:raise ValueError('Search initial endpoint belongs to another checkpoint')
    validate_profile(initial)
    if type(selection_draw) not in (int,float) or not 0<=selection_draw<1:raise ValueError('Selection draw must be frozen before training')
    r=cfg['search_round']
    if cfg['protocol_version']!=version(r):raise ValueError('Search preparation requires the whole round config')
    # Structural candidate records adapt the existing selector interface. These
    # are deterministic target-only branch roles, not generated curricula.
    return {'config':copy.deepcopy(cfg),'initial':copy.deepcopy(initial),'selection_draw':selection_draw,
        'reward_condition':initial['target']['bottleneck_index'],
        'generation':{'source':'deterministic_target_only_branch_roles','model_generated':False,
            'candidates':[{'index':i,'valid':True,'role':'target_only_search'} for i in (1,2,3)]},
        'branches':[{'index':i,'version':version(r,i),'solver_start':cfg['solver_start'],
            'kind':'target','reward_mode':'binary','tokens':524288} for i in range(4)],
        'search_protocol_frozen':True,'suite_complete':False}


def select_round(frozen,branches):
    """Reuse the primary selector; training activity never estimates endpoint rates."""
    from verge_round_decision import decide
    cfg=frozen['config'];r=cfg['search_round']
    validate_round_config(cfg)
    if (set(branches)!=set(range(4)) or cfg.get('target_only_search') is not True
            or cfg['protocol_version']!=version(r) or cfg.get('book_suite')
            or frozen.get('search_protocol_frozen') is not True):
        raise ValueError('Need all four isolated target-only branches')
    validate_profile(frozen['initial'])
    for i,b in branches.items():
        validate_profile(b);state=b['training'];stem=f'checkpoints/{version(r,i)}_direct/resume_u'
        if (state['train_tokens']!=524288 or type(state['update']) is not int or not 0<state['update']<=100
                or type(state['optimizer_steps']) is not int or not 0<=state['optimizer_steps']<=state['update']
                or b['checkpoint']!=stem+f"{state['update']:04d}"):
            raise ValueError('Incomplete or foreign target-only branch result')
    context=copy.deepcopy(frozen)
    if branches[0]['training']['optimizer_steps']==0:
        # The primary book uses exactly zero start-vs-direct gain when a fresh
        # zero-decay direct optimizer never stepped. Normalize only the selector
        # input for this same identity rule; retain the actual initial profile.
        context['initial']['target']=copy.deepcopy(branches[0]['target'])
    decision=decide(context,branches)
    decision.update(target_condition_gains=decision['challenger_rewards'],
        challenger_update_allowed=False,challenger_update_performed=False,
        branch_roles='Four target-only branches; indices 1-3 are not curriculum proposals',
        search_round=r,protocol_version=version(r),search_round_complete=False,suite_complete=False,
        observed_initial_profile_retained=True,selector='Existing VERGE endpoint selector with the same zero-update reference identity rule')
    return decision
