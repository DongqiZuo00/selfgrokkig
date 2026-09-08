"""Launch and metadata-copy contracts for isolated target-only search workers."""
import copy
import json
import re
from pathlib import Path
from verge_search_protocol import PLAN,SUITE,round_config,worker_config,validate_plan,version,prepared_round

PATTERN=r'verge_search_v1_r0[0-5]_b[0-3]'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def worker_prepared(cfg,initial):
    if (not re.fullmatch(PATTERN,cfg['protocol_version']) or cfg.get('target_only_search') is not True
            or initial['checkpoint']!=cfg['solver_start']):raise ValueError('Foreign search worker or initial endpoint')
    return {'config':copy.deepcopy(cfg),'initial':copy.deepcopy(initial),'target_only_search':True,
        'initial_endpoint_reused_from_round':True,'additional_initial_model_draws':0,
        'branches':[{'index':0,'id':cfg['protocol_version']+'_direct',
            'stages':[dict(p,path=cfg['target_train']) for p in cfg['search_phases']]}]}


def validate_prepared(cfg,frozen):
    if frozen!=worker_prepared(cfg,frozen['initial']):raise RuntimeError('Search preparation changed')
    return frozen


def validate_round_launch(exp,cfg,*,require_prepared=False):
    exp=Path(exp).resolve();root=exp/'raw_results'/SUITE
    if (not re.fullmatch(r'verge_search_v1_r0[0-5]',cfg.get('protocol_version','')) or cfg.get('target_only_search') is not True
            or cfg.get('book_suite') or cfg.get('control_only') or cfg.get('followon_only')):
        raise RuntimeError('Only the isolated search round namespace is allowed')
    plan=validate_plan(read(exp/PLAN))
    if read(root/'protocol_frozen.json')!=plan:raise RuntimeError('Search plan differs from submitted plan')
    predecessors=[('verge_book_v2','PRIMARY_BLOCK_COMPLETE.json','primary_block_complete',('outer_rounds',24)),
        ('verge_control_v1','DIRECT_CONTROL_BLOCK_COMPLETE.json','control_block_complete',('segments',18)),
        ('verge_followon_v1','FOLLOWON_BLOCK_COMPLETE.json','followon_block_complete',('candidates',72))]
    for suite,file,flag,(count,n) in predecessors:
        record=read(exp/'raw_results'/suite/file)
        if record.get(flag) is not True or record.get(count)!=n or record.get('suite_complete') is not False:
            raise RuntimeError('A required predecessor block is incomplete')
    r=cfg['search_round'];name=version(r)
    previous=read(root/'rounds'/version(r-1)/'complete.json') if r else None
    expected_round=round_config(exp,r,plan,previous)
    if read(exp/'manifests'/(name+'.json'))!=expected_round or cfg!=expected_round:
        raise RuntimeError('Search round differs from its frozen configuration or predecessor')
    source=(exp/cfg['solver_start']).resolve()
    if not source.is_relative_to((exp/'checkpoints').resolve()):raise RuntimeError('Search checkpoint escaped experiment')
    for file in ('verge_committed.json','adapter_model.safetensors','state.pt'):
        if not (source/file).is_file() or not (source/file).stat().st_size:raise RuntimeError('Missing committed search source')
    if read(source/'verge_committed.json')['update']!=int(source.name.removeprefix('resume_u')):
        raise RuntimeError('Search source checkpoint is not committed')
    if require_prepared:
        frozen=read(root/'rounds'/name/'round_frozen.json')
        if frozen!=prepared_round(expected_round,frozen['initial'],frozen['selection_draw']):
            raise RuntimeError('Shared initial endpoint or selector draw changed')
        return frozen
    return None


def validate_launch(exp,cfg):
    exp=Path(exp).resolve()
    if (not re.fullmatch(PATTERN,cfg.get('protocol_version','')) or cfg.get('target_only_search') is not True
            or cfg.get('book_suite') or cfg.get('control_only') or cfg.get('followon_only')):
        raise RuntimeError('Only the isolated search worker namespace is allowed')
    round_cfg=read(exp/'manifests'/(version(cfg['search_round'])+'.json'))
    frozen=validate_round_launch(exp,round_cfg,require_prepared=True)
    if cfg!=worker_config(round_cfg,cfg['search_branch_index']):
        raise RuntimeError('Search worker differs from the frozen round or branch stream')
    return frozen
