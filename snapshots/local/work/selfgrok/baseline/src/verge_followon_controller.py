"""One bounded 72-candidate array after completed primary/direct-control blocks.

Each element prepares existing metadata, trains one continuation and releases one
B200. Submission intents prevent blind duplicate allocations after lost replies.
This controller never reports the experiment book or follow-on analysis complete.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from verge_control_controller import read,write
from verge_followon_protocol import ARMS,candidate_config,validate_plan,validate_launch,validate_prepared

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP/'raw_results/verge_followon_v1'
PLAN='manifests/verge_followon_v1_protocol.json'


def version(index):
    if type(index) is not int or not 0<=index<72:raise ValueError('Invalid follow-on array index')
    return f'verge_followon_v1_{ARMS[index%12//3]}_r{index//12:02d}_c{index%3+1}'


def predecessor_finished():
    primary=EXP/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json'
    control=EXP/'raw_results/verge_control_v1/DIRECT_CONTROL_BLOCK_COMPLETE.json'
    if (not primary.is_file() or read(primary).get('primary_block_complete') is not True
            or read(primary).get('outer_rounds')!=24 or not control.is_file()
            or read(control).get('control_block_complete') is not True):
        raise RuntimeError('Primary and direct-control predecessors must finish before follow-on dispatch')
    last=read(EXP/'raw_results/verge_control_v1/jobs/round_05.json')['coordinator']
    if not isinstance(last,str) or not last.isdigit():raise RuntimeError('Missing final predecessor dependency')
    return last


def analysis_strata(bank):
    seen={arm:False for arm in ARMS};strata=[]
    for cohort in bank['cohorts']:
        first=cohort['candidates'][0];arm=first['source_arm']
        clear=cohort.get('no_full_event_recorded_in_source_cohort')
        if type(clear) is not bool:raise RuntimeError('Source event context is missing')
        strata.append({'source_cohort':cohort['cohort'],'source_arm':arm,'source_round':first['source_round'],
            'source_solver_initial':cohort['source_solver_initial'],
            'earlier_same_arm_source_event_recorded':seen[arm],
            'pre_success_analysis':not seen[arm] and clear,'candidates_retained':3})
        seen[arm]=seen[arm] or not clear
    return {'cohorts':strata,'candidate_filter_applied':False,
            'defined_before_followon_training':True,'based_on_followon_outcomes':False}


@contextmanager
def dispatch_lock():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);yield


def keep_immutable(path,value):
    if path.exists() and read(path)!=value:raise RuntimeError(f'Frozen record changed: {path.name}')
    if not path.exists():write(path,value)


def submit(stage,dependency):
    if stage not in ('workers','coordinator') or not isinstance(dependency,str) or not dependency.isdigit():
        raise ValueError('An explicit stage and exact prior-job dependency are required')
    args=['sbatch','--parsable',f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes',
        '--export=ALL,VERGE_BOOK_SUITE=verge_book_v2,VERGE_FOLLOWON_SUITE=verge_followon_v1',
        f'--job-name=vfo-{stage}']
    if stage=='workers':args+=['--array=0-71%2',str(EXP/'scripts/verge_followon_workers.sbatch')]
    else:args+=[str(EXP/'scripts/verge_followon_coordinate.sbatch')]
    intent=ROOT/'submission_intents'/f'{stage}.json'
    if intent.exists():
        record=read(intent)
        if record.get('arguments')!=args:raise RuntimeError('Submission intent arguments changed')
        job=record.get('confirmed_job_id')
        if isinstance(job,str) and job.isdigit():return job
        raise RuntimeError('Unresolved previous submission; reconcile Slurm before retrying')
    write(intent,{'arguments':args,'created_at':time.time(),'confirmed_job_id':None})
    response=subprocess.run(args,cwd=EXP.parents[1],text=True,capture_output=True,check=True)
    job=response.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain sbatch response; no blind retry')
    write(intent,{'arguments':args,'confirmed_job_id':job,'confirmed_at':time.time()})
    return job


def completed(index,*,deep=False):
    name=version(index);directory=EXP/'raw_results'/name
    path=directory/'branches/0/complete.json'
    if not path.is_file():return None
    cfg=read(EXP/'manifests'/f'{name}.json');validate_launch(EXP,cfg)
    frozen=read(directory/'round_frozen.json');validate_prepared(cfg,frozen)
    result=read(path);state=result['training'];checkpoint=result.get('checkpoint','')
    if (result.get('followon_segment_verified') is not True or result.get('followon_only') is not True
            or result.get('followon_comparison_complete') is not False or result.get('suite_complete') is not False
            or state['train_tokens']!=262144 or not 0<state['update']<=100
            or checkpoint!=f"checkpoints/{name}_direct/resume_u{state['update']:04d}"
            or result.get('source_candidate')!=cfg['followon_source']):
        raise RuntimeError('Invalid follow-on completion record')
    cp=(EXP/checkpoint).resolve()
    if not cp.is_relative_to((EXP/'checkpoints').resolve()):raise RuntimeError('Checkpoint escaped experiment')
    paths=[cp/f for f in ('verge_committed.json','adapter_model.safetensors','state.pt')]
    paths += [directory/'branches/0'/f for f in ('target.jsonl','scope.jsonl','completion.jsonl','completion_profile.json')]
    if any(not p.is_file() or p.stat().st_size==0 for p in paths):raise RuntimeError('Missing follow-on output')
    if read(cp/'verge_committed.json')['update']!=state['update']:raise RuntimeError('Uncommitted follow-on endpoint')
    if deep:
        from verge_followon_verify import validate_segment
        validate_segment(frozen,result,directory/'branches/0')
    return result


def dispatch():
    if os.environ.get('VERGE_BOOK_SUITE')!='verge_book_v2' or os.environ.get('VERGE_FOLLOWON_SUITE')!='verge_followon_v1':
        raise RuntimeError('Both explicit suite variables are required')
    dependency=predecessor_finished();plan=validate_plan(read(EXP/PLAN))
    with dispatch_lock():
        keep_immutable(ROOT/'protocol_frozen.json',plan)
        bank_path=ROOT/'source_bank.json'
        if bank_path.exists():bank=read(bank_path)
        else:
            from verge_followon_sources import collect
            bank=collect();write(bank_path,bank)
        keep_immutable(ROOT/'source_bank_frozen.json',bank)
        # Complete deterministic coverage is checked before any sbatch command.
        for i in range(72):
            cfg=candidate_config(EXP,bank,i,plan)
            if cfg['protocol_version']!=version(i):raise RuntimeError('Array and candidate mapping differ')
            validate_launch(EXP,cfg)
            keep_immutable(EXP/'manifests'/f'{version(i)}.json',cfg)
        keep_immutable(ROOT/'cohort_analysis_strata.json',analysis_strata(bank))
        jobs_path=ROOT/'jobs/block.json'
        jobs=read(jobs_path) if jobs_path.exists() else {'candidate_count':72,'indices':list(range(72))}
        if jobs.get('candidate_count')!=72 or jobs.get('indices')!=list(range(72)):
            raise RuntimeError('Changed all-candidate allocation registry')
        if 'workers' not in jobs:
            jobs['workers']=submit('workers',dependency);write(jobs_path,jobs)
        if 'coordinator' not in jobs:
            jobs['coordinator']=submit('coordinator',jobs['workers']);write(jobs_path,jobs)
        write(ROOT/'active_wave.json',jobs)
        return jobs


def coordinate():
    for i in range(72):
        if completed(i,deep=True) is None:raise RuntimeError('Do not finish an incomplete all-candidate array')
    result={'all_candidate_followons_finished':True,'candidates':72,'candidate_loss_tokens':18874368,
            'followon_block_complete':False,'suite_complete':False,'completed_at':time.time(),
            'remaining':'Censoring/ranking analysis, resource accounting, figures, CSV, LaTeX and artifact validation'}
    write(ROOT/'FOLLOWONS_FINISHED.json',result)
    return result


def status():
    return {'completed_candidates':[i for i in range(72) if completed(i)],'required_candidates':72,
        'active_wave':read(ROOT/'active_wave.json') if (ROOT/'active_wave.json').exists() else None,
        'followon_block_complete':False,'suite_complete':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('dispatch','coordinate','status'))
    action=p.parse_args().action
    print(json.dumps({'dispatch':dispatch,'coordinate':coordinate,'status':status}[action](),indent=2))
