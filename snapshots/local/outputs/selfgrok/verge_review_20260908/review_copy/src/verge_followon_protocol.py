"""Separate, bounded all-candidate continuation configuration; no submission."""
import copy
import json
from pathlib import Path
import re
from verge_repair_config import repair_config

ARMS = ('verge','frozen','uncertainty','outcome')
PATTERN = r'verge_followon_v1_(verge|frozen|uncertainty|outcome)_r0[0-5]_c[1-3]'


def validate_plan(plan):
    expected = {'protocol':'verge_followon_v1_all_candidates','source_suite':'verge_book_v2',
        'backbone':'mistralai/Ministral-3-3B-Instruct-2512-BF16','candidate_count':72,
        'loss_tokens_per_candidate':262144,'total_candidate_loss_tokens':18874368,
        'training_seed':42,'solver_reward':'binary_complete_target_verifier_only',
        'optimizer_reset':'fresh_per_followon_candidate','solver_learning_rate':1e-5,
        'optimizer_betas':[.9,.95],'weight_decay':0.,'kl_beta':.02,
        'kl_reference':'own_source_candidate_endpoint','maximum_updates_per_branch':100,
        'completion_tokens':2048,'group_size':8,'token_update_threshold':16384,
        'temperature':1.,'top_p':1.,'top_k':-1,
        'sampling_protocol':'per_prompt_disjoint_seed_ranges_v1',
        'early_stop':False,'adaptive_checkpoint_selection':False,'requires_primary_complete':True,
        'execute_after_block':'verge_control_v1','maximum_total_gpus':2,
        'maximum_total_cpu_memory_gb':64,'extra_audit_or_screening_samples':0,
        'checkpoint_hash_scans':0,'official_test_opened':False,'total_compute_matched':False,
        'book_complete_after_this_block':False}
    if any(plan.get(k) != v for k,v in expected.items()):
        raise ValueError('Changed or unbounded all-candidate continuation plan')
    if plan.get('final_endpoint') != {'target_instances':64,'samples_per_instance':8,
            'scope_instances':32,'scope_samples':4,'completion_instances':64,'completion_samples':1}:
        raise ValueError('Changed follow-on endpoint budget')
    if plan.get('worker') != {'gpus':1,'gpu_type':'B200','cpus':8,'cpu_memory_gb':32,'wall_hours':2}:
        raise ValueError('Changed follow-on worker allocation')
    if plan.get('coordinator') != {'gpus':0,'cpus':2,'cpu_memory_gb':4,'after_all_workers_exit':True}:
        raise ValueError('Coordinator must wait until all GPU workers exit')
    required_notes = ('event_clock','censoring','pre_success_analysis','ranking_strata','condition_score',
                      'uncertainty_score','tie_rule','random_reference','rank_statistics','inference_limit')
    if any(not isinstance(plan.get(k),str) or not plan[k].strip() for k in required_notes):
        raise ValueError('Event, censoring and ranking choices must be explicit before submission')
    return plan


def candidate_from_bank(bank, index):
    if (type(index) is not int or not 0 <= index < 72 or bank.get('candidate_count') != 72
            or bank.get('protocol') != 'verge_followon_v1_source_bank'
            or len(bank.get('cohorts',[])) != 24):
        raise ValueError('A complete 72-candidate source bank and bounded array index are required')
    sequence = []
    for r in range(6):
        for arm in ARMS:
            cohort = bank['cohorts'][len(sequence)//3]
            name = f'verge_book_v2_{arm}_r{r:02d}'
            if cohort['cohort'] != name or [c['candidate_index'] for c in cohort['candidates']] != [1,2,3]:
                raise ValueError('Source bank is missing, duplicated or reordered')
            for c in cohort['candidates']:
                if (c['source_cohort'] != name or c['source_arm'] != arm or c['source_round'] != r
                        or c['follow_on_loss_tokens'] != 262144
                        or c['source_solver_initial'] != cohort['source_solver_initial']):
                    raise ValueError('Candidate source and cohort context disagree')
                sequence.append(c)
    return copy.deepcopy(sequence[index])


def candidate_config(exp, bank, index, plan):
    validate_plan(plan)
    c = candidate_from_bank(bank,index)
    arm,r,i = c['source_arm'],c['source_round'],c['candidate_index']
    source_pattern = rf'checkpoints/verge_book_v2_{arm}_r{r:02d}_candidate_{i}/resume_u\d{{4}}'
    if not re.fullmatch(source_pattern,c['source_checkpoint']):
        raise ValueError('Follow-on cannot silently switch the source candidate checkpoint')
    cfg = repair_config('verge_mistral_repair_v2',Path(exp))
    name = f'verge_followon_v1_{arm}_r{r:02d}_c{i}'
    cfg.update(protocol_version=name, followon_only=True, followon_source=copy.deepcopy(c),
        followon_array_index=index, followon_block_protocol='manifests/verge_followon_v1_protocol.json',
        followon_phases=[{'kind':'target','reward_mode':'binary','tokens':262144}],
        backbone=plan['backbone'], solver_start=c['source_checkpoint'], solver_rank=16,
        solver_learning_rate=1e-5, weight_decay=0., kl_beta=.02, seed=42,
        fresh_optimizer=True, maximum_updates_per_branch=100, train_tokens_per_branch=262144,
        alpha=1., eta=1., outer_rounds=1, proposal_count=0, challenger_start=None,
        stage_probe_instances=0, token_update_threshold=16384,
        selection_samples_per_instance=8, endpoint_instances=64, scope_instances=32,
        scope_samples=4, completion_instances=64, completion_samples=1, completion_tokens=2048,
        independent_prompt_streams=True, sampling_protocol='per_prompt_disjoint_seed_ranges_v1',
        random_stream_id=f'verge_followon_v1_paired_{arm}_r{r:02d}',
        final_evaluation_allowed=False, official_test_opened=False, stop_after_first_success=False,
        adaptive_checkpoint_selection=False, acceptance_only=False,
        scope='One fixed-budget target-only continuation of one original candidate; no Challenger or selector',
        budget_note='Equal 262144 charged loss tokens per original candidate, not matched total compute',
        explicit_pilot_differences=[
            'User-selected Mistral backbone, one training seed42 and fixed 2048 native-token completion cap',
            'Fresh optimizer and source-candidate KL reference define the same continuation intervention for every candidate',
            'No new initial evaluation or candidate screening; saved source endpoint is metadata only',
            'Target selection/scope/completion sampled only at the final fixed checkpoint; no official test access',
            'No event is right-censored, not proof of impossible learning; no iid-branch or cross-seed confidence claims'])
    return cfg


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepared_round(cfg, initial):
    if initial['checkpoint'] != cfg['solver_start']:
        raise ValueError('Saved source endpoint belongs to another checkpoint')
    return {'config':cfg,'initial':copy.deepcopy(initial),'followon_only':True,
        'initial_endpoint_reused_from_source':True,'initial_model_draws':0,
        'branches':[{'index':0,'id':cfg['protocol_version']+'_direct',
                     'stages':[dict(p,path=cfg['target_train']) for p in cfg['followon_phases']]}]}


def validate_prepared(cfg, frozen):
    if frozen != prepared_round(cfg,frozen['initial']):
        raise RuntimeError('Follow-on preparation differs from its fixed source and configuration')
    return frozen


def validate_launch(exp, cfg):
    """Not called by the shared runtime until the separate runner is implemented."""
    exp = Path(exp).resolve()
    name = cfg.get('protocol_version','')
    if (not re.fullmatch(PATTERN,name) or cfg.get('followon_only') is not True
            or cfg.get('book_suite') or cfg.get('control_only')):
        raise RuntimeError('Foreign namespace is not a follow-on continuation')
    root = exp/'raw_results/verge_followon_v1'
    paths = [exp/'manifests/verge_followon_v1_protocol.json',root/'protocol_frozen.json',
             root/'source_bank.json',root/'source_bank_frozen.json']
    if any(not p.is_file() for p in paths):
        raise RuntimeError('Complete source bank and submission-time frozen plan are required')
    plan = validate_plan(read(paths[0]))
    bank = read(paths[2])
    if read(paths[1]) != plan or read(paths[3]) != bank:
        raise RuntimeError('Source bank or continuation plan changed after submission')
    primary = exp/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json'
    control = exp/'raw_results/verge_control_v1/DIRECT_CONTROL_BLOCK_COMPLETE.json'
    if (not primary.is_file() or read(primary).get('primary_block_complete') is not True
            or read(primary).get('outer_rounds') != 24 or not control.is_file()
            or read(control).get('control_block_complete') is not True):
        raise RuntimeError('Required predecessor blocks have not completed')
    expected = candidate_config(exp,bank,cfg['followon_array_index'],plan)
    if cfg != expected:
        raise RuntimeError('Candidate configuration differs from the frozen intervention')
    source = (exp/cfg['solver_start']).resolve()
    if not source.is_relative_to((exp/'checkpoints').resolve()):
        raise RuntimeError('Source checkpoint escaped the experiment')
    for filename in ('verge_committed.json','adapter_model.safetensors','state.pt'):
        if not (source/filename).is_file() or (source/filename).stat().st_size == 0:
            raise RuntimeError('Source candidate checkpoint is not available')
    if read(source/'verge_committed.json')['update'] != int(source.name.removeprefix('resume_u')):
        raise RuntimeError('Source checkpoint update is not committed')
    return plan
