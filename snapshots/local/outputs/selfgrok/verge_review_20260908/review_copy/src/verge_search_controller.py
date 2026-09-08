"""Six bounded four-branch search waves after verified follow-on completion.

Submissions use durable intents. Every prepare and finish owns one B200; only
the worker array overlaps two one-B200 tasks. This module does not enable the
shared runtime and cannot declare the whole search comparison complete.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time
from verge_control_controller import read,write
from verge_search_protocol import PLAN,SUITE,round_config,worker_config,validate_plan,version,select_round
from verge_search_worker_protocol import validate_round_launch,validate_launch,validate_prepared

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP/'raw_results'/SUITE


def predecessor_finished():
    for suite,file,flag,count,n in (
        ('verge_book_v2','PRIMARY_BLOCK_COMPLETE.json','primary_block_complete','outer_rounds',24),
        ('verge_control_v1','DIRECT_CONTROL_BLOCK_COMPLETE.json','control_block_complete','segments',18),
        ('verge_followon_v1','FOLLOWON_BLOCK_COMPLETE.json','followon_block_complete','candidates',72)):
        path=EXP/'raw_results'/suite/file
        if not path.is_file():raise RuntimeError('Required predecessor block is incomplete')
        record=read(path)
        if record.get(flag) is not True or record.get(count)!=n or record.get('suite_complete') is not False:
            raise RuntimeError('Required predecessor block is incomplete')
    last=read(EXP/'raw_results/verge_followon_v1/jobs/block.json')['coordinator']
    if not isinstance(last,str) or not last.isdigit():raise RuntimeError('Missing exact final predecessor dependency')
    return last


def keep(path,value):
    if path.exists() and read(path)!=value:raise RuntimeError('Frozen search record changed: '+path.name)
    if not path.exists():write(path,value)


@contextmanager
def dispatch_lock():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);yield


def submit(stage,r,dependency):
    version(r)
    if stage not in ('prepare','workers','finish') or not isinstance(dependency,str) or not dependency.isdigit():
        raise ValueError('A bounded search stage and exact predecessor are required')
    args=['sbatch','--parsable',f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes',
        '--export=ALL,VERGE_BOOK_SUITE=verge_book_v2,VERGE_SEARCH_SUITE=verge_search_v1',
        f'--job-name=vs-r{r}-{stage}']
    if stage=='workers':args+=['--array=0-3%2',str(EXP/'scripts/verge_search_workers.sbatch'),str(r)]
    else:args+=[str(EXP/'scripts/verge_search_round.sbatch'),str(r),stage]
    intent=ROOT/'submission_intents'/f'round_{r:02d}_{stage}.json'
    if intent.exists():
        saved=read(intent)
        if saved.get('arguments')!=args:raise RuntimeError('Submission arguments changed after intent')
        job=saved.get('confirmed_job_id')
        if isinstance(job,str) and job.isdigit():return job
        raise RuntimeError('Unresolved submission; reconcile Slurm before retrying')
    write(intent,{'arguments':args,'created_at':time.time(),'confirmed_job_id':None})
    response=subprocess.run(args,cwd=EXP.parents[1],text=True,capture_output=True,check=True)
    job=response.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Uncertain sbatch response; no blind retry')
    write(intent,{'arguments':args,'confirmed_job_id':job,'confirmed_at':time.time()})
    return job


def completed_worker(r,b,*,deep=False):
    name=version(r,b);directory=EXP/'raw_results'/name;output=directory/'branches/0'
    if not (output/'complete.json').exists():return None
    cfg=read(EXP/'manifests'/(name+'.json'));shared=validate_launch(EXP,cfg)
    frozen=read(directory/'round_frozen.json');validate_prepared(cfg,frozen)
    if frozen['initial']!=shared['initial']:raise RuntimeError('Worker substituted the shared initial endpoint')
    result=read(output/'complete.json');state=result['training']
    checkpoint=f"checkpoints/{name}_direct/resume_u{state['update']:04d}"
    if (result.get('search_segment_verified') is not True or result.get('target_only_search') is not True
            or result.get('search_comparison_complete') is not False or result.get('suite_complete') is not False
            or result.get('search_round')!=r or result.get('search_branch_index')!=b
            or state['train_tokens']!=524288 or type(state['update']) is not int or not 0<state['update']<=100
            or result.get('checkpoint')!=checkpoint):raise RuntimeError('Invalid search worker completion')
    source=(EXP/checkpoint).resolve()
    if not source.is_relative_to((EXP/'checkpoints').resolve()):raise RuntimeError('Search checkpoint escaped experiment')
    files=[source/f for f in ('state.pt','adapter_model.safetensors','verge_committed.json')]
    files += [output/f for f in ('target.jsonl','scope.jsonl','target.response.json','scope.response.json')]
    if any(not p.is_file() or not p.stat().st_size for p in files):raise RuntimeError('Missing search worker artifact')
    if read(source/'verge_committed.json')['update']!=state['update']:raise RuntimeError('Uncommitted search endpoint')
    if deep:
        from verge_search_verify import validate_segment
        validate_segment(frozen,result,output)
    return result


def completed_round(r,*,deep=False):
    name=version(r);directory=ROOT/'rounds'/name;path=directory/'complete.json'
    if not path.exists():return None
    cfg=read(EXP/'manifests'/(name+'.json'));frozen=validate_round_launch(EXP,cfg,require_prepared=True)
    record=read(path)
    if (record.get('protocol_version')!=name or record.get('search_round_complete') is not True
            or record.get('search_block_complete') is not False or record.get('suite_complete') is not False
            or record.get('completed_branches')!=4 or record.get('round_loss_tokens')!=4*524288
            or record.get('official_test_opened') is not False):raise RuntimeError('Invalid search round completion')
    results={b:completed_worker(r,b,deep=deep) for b in range(4)}
    if any(v is None for v in results.values()):raise RuntimeError('A search round cannot skip a worker')
    expected=select_round(frozen,results)
    if read(directory/'decision.json')!=expected or record['selected']!=expected['selected']:
        raise RuntimeError('Selection differs from frozen endpoint decision')
    for file in ('completion_profile.json','completion.jsonl','completion.response.json'):
        p=directory/file
        if not p.is_file() or not p.stat().st_size:raise RuntimeError('Missing selected-checkpoint completion evaluation')
    completion=read(directory/'completion_profile.json')
    if (record.get('completion_draws')!=64 or completion.get('rollouts')!=64
            or record.get('completion_full_successes')!=completion['counts'][-1]
            or completion.get('checkpoint')!=expected['selected']['checkpoint']
            or record.get('completion_used_for_selection') is not False):
        raise RuntimeError('Selected completion metadata differs from the original evaluation')
    if deep:
        from verge_search_round_verify import validate_initial,validate_completion
        validate_initial(cfg,frozen['initial'],directory/'initial')
        validate_completion(cfg,expected,completion,directory)
    return record


def dispatch():
    if os.environ.get('VERGE_BOOK_SUITE')!='verge_book_v2' or os.environ.get('VERGE_SEARCH_SUITE')!=SUITE:
        raise RuntimeError('Both explicit suite variables are required')
    dependency=predecessor_finished();plan=validate_plan(read(EXP/PLAN))
    ready=ROOT/'IMPLEMENTATION_READY.json'
    if not ready.is_file() or any(read(ready).get(k) is not True for k in (
            'runtime_enabled','worker_tests_passed','round_tests_passed','controller_tests_passed')):
        raise RuntimeError('Search runtime integration and tests are not ready for submission')
    with dispatch_lock():
        keep(ROOT/'protocol_frozen.json',plan)
        previous=None
        for r in range(6):
            completed=completed_round(r)
            jobs_path=ROOT/'jobs'/f'round_{r:02d}.json'
            if completed is not None:
                previous=completed;dependency=read(jobs_path)['finish'];continue
            cfg=round_config(EXP,r,plan,previous)
            keep(EXP/'manifests'/(version(r)+'.json'),cfg)
            validate_round_launch(EXP,cfg)
            for b in range(4):keep(EXP/'manifests'/(version(r,b)+'.json'),worker_config(cfg,b))
            jobs=read(jobs_path) if jobs_path.exists() else {'round':r,'version':version(r),'branches':[0,1,2,3]}
            if jobs.get('round')!=r or jobs.get('version')!=version(r) or jobs.get('branches')!=[0,1,2,3]:
                raise RuntimeError('Search job registry changed')
            for stage in ('prepare','workers','finish'):
                if stage not in jobs:
                    jobs[stage]=submit(stage,r,dependency);write(jobs_path,jobs)
                dependency=jobs[stage]
            write(ROOT/'active_wave.json',jobs);return jobs
        result={'search_rounds_finished':True,'rounds':6,'branches':24,'loss_tokens':12582912,
            'search_block_complete':False,'suite_complete':False,'completed_at':time.time(),
            'remaining':'Actual endpoint/event analysis, resource ledger, figures, CSV, LaTeX and artifact validation'}
        write(ROOT/'SEARCH_ROUNDS_FINISHED.json',result);return result


def coordinate(r):
    version(r)
    if completed_round(r,deep=True) is None:raise RuntimeError('Do not advance an incomplete search round')
    return dispatch()


def status():
    return {'completed_rounds':[r for r in range(6) if completed_round(r) is not None],
        'required_rounds':6,'active_wave':read(ROOT/'active_wave.json') if (ROOT/'active_wave.json').exists() else None,
        'search_block_complete':False,'suite_complete':False}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('dispatch','coordinate','status'))
    p.add_argument('--round',type=int);a=p.parse_args()
    result=dispatch() if a.action=='dispatch' else coordinate(a.round) if a.action=='coordinate' else status()
    print(json.dumps(result,indent=2))
