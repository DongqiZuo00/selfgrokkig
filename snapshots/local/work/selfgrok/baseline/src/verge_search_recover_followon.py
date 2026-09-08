"""One journaled recovery of the expired follow-on -> search dependency.

Only the first prepare dependency may be released. A confirmed successful
predecessor, rejected original submission, empty search history and no in-scope
live allocation are all required. This is not a generic resubmission helper.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import verge_search_controller as ctl

DEPENDENCY = '41333225'


def expected_arguments(exp):
    return ['sbatch', '--parsable', '--dependency=afterok:' + DEPENDENCY,
            '--kill-on-invalid-dep=yes',
            '--export=ALL,VERGE_BOOK_SUITE=verge_book_v2,VERGE_SEARCH_SUITE=verge_search_v1',
            '--job-name=vs-r0-prepare', str(exp/'scripts/verge_search_round.sbatch'), '0', 'prepare']


def released_arguments(original, rejected, expected, accounting, live_returncode,
                       live_stderr, queue, matching_history, work_root):
    if (original.get('arguments') != expected or original.get('confirmed_job_id') is not None
            or rejected.get('arguments') != expected or rejected.get('returncode') != 1
            or rejected.get('stdout', '').strip()
            or 'Batch job submission failed: Job dependency problem' not in rejected.get('stderr', '')):
        raise RuntimeError('Original submission is not a certified dependency rejection')
    rows = [line.strip().split('|') for line in accounting.splitlines() if line.strip()]
    if [r for r in rows if r[0] == DEPENDENCY] != [[DEPENDENCY, 'COMPLETED', '0:0']]:
        raise RuntimeError('Exact follow-on coordinator did not complete successfully')
    if live_returncode != 1 or live_stderr.strip() != 'slurm_load_jobs error: Invalid job id specified':
        raise RuntimeError('Predecessor purge is not confirmed')
    if matching_history.strip():
        raise RuntimeError('Possible earlier search allocation; reconcile instead of resubmitting')
    for line in queue.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 3:
            raise RuntimeError('Cannot determine live allocation ownership')
        _, name, directory = fields
        path = Path(directory)
        if (name.startswith(('vs-', 'verge-search')) or not path.is_absolute()
                or path.resolve().is_relative_to(work_root.resolve())):
            raise RuntimeError('An in-scope or unidentified live allocation blocks recovery')
    needle = '--dependency=afterok:' + DEPENDENCY
    if expected.count(needle) != 1:
        raise RuntimeError('Ambiguous dependency')
    return [a for a in expected if a != needle]


def command(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def recover():
    if (os.environ.get('VERGE_BOOK_SUITE') != 'verge_book_v2'
            or os.environ.get('VERGE_SEARCH_SUITE') != 'verge_search_v1'):
        raise RuntimeError('Both suite variables must be explicit')
    with ctl.dispatch_lock():
        if ctl.predecessor_finished() != DEPENDENCY:
            raise RuntimeError('This helper is only for the original follow-on coordinator')
        jobs_path = ctl.ROOT/'jobs/round_00.json'
        replacement = ctl.ROOT/'submission_intents/round_00_prepare_expired_followon_recovery.json'
        if (jobs_path.exists() or replacement.exists() or (ctl.ROOT/'active_wave.json').exists()
                or any((ctl.ROOT/'jobs').glob('round_*.json'))):
            raise RuntimeError('Recovery already attempted or search registered; no blind retry')
        intent_path = ctl.ROOT/'submission_intents/round_00_prepare.json'
        original = ctl.read(intent_path)
        rejected = ctl.read(ctl.ROOT/'submission_intents/first_dispatch_rejection.json')
        if ctl.read(ctl.ROOT/'protocol_frozen.json') != ctl.validate_plan(ctl.read(ctl.EXP/ctl.PLAN)):
            raise RuntimeError('Search plan changed after the original submission')
        ready = ctl.read(ctl.ROOT/'IMPLEMENTATION_READY.json')
        if any(ready.get(k) is not True for k in ('runtime_enabled', 'worker_tests_passed',
                                                 'round_tests_passed', 'controller_tests_passed')):
            raise RuntimeError('Runtime integration is not ready')
        cfg = ctl.read(ctl.EXP/'manifests'/f'{ctl.version(0)}.json')
        ctl.validate_round_launch(ctl.EXP, cfg)
        for b in range(4):
            if ctl.read(ctl.EXP/'manifests'/f'{ctl.version(0,b)}.json') != ctl.worker_config(cfg,b):
                raise RuntimeError('First-round worker configuration changed')
        account = command(['sacct', '-j', DEPENDENCY, '--format=JobIDRaw,State,ExitCode', '-n', '-P'], check=True)
        live = command(['scontrol', 'show', 'job', DEPENDENCY])
        queue = command(['squeue', '-h', '-u', 'jinjiaguo', '-o', '%i|%j|%Z'], check=True)
        history = command(['sacct', '-u', 'jinjiaguo', '-S', '2026-09-07T00:00:00',
                           '--name=vs-r0-prepare,vs-r0-workers,vs-r0-finish',
                           '--format=JobIDRaw,State,ExitCode', '-n', '-P'], check=True)
        args = released_arguments(original, rejected, expected_arguments(ctl.EXP), account.stdout,
                                  live.returncode, live.stderr, queue.stdout, history.stdout, ctl.EXP.parents[1])
        test = command([args[0], '--test-only', *args[1:]], cwd=ctl.EXP.parents[1])
        if test.returncode:
            raise RuntimeError('Replacement test-only rejected: ' + test.stderr)
        receipt = dict(observed_at=time.time(), reason='Successful follow-on coordinator purged from Slurm',
                       predecessor=DEPENDENCY, accounting=account.stdout,
                       live_returncode=live.returncode, live_stderr=live.stderr,
                       queue=queue.stdout, matching_search_history=history.stdout,
                       original_intent=str(intent_path.relative_to(ctl.EXP)), original_intent_preserved=True,
                       original_rejection=rejected, test_only_stdout=test.stdout, test_only_stderr=test.stderr,
                       arguments=args, confirmed_job_id=None, scientific_protocol_changed=False)
        ctl.write(replacement, receipt)
        response = command(args, cwd=ctl.EXP.parents[1])
        receipt.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
        ctl.write(replacement, receipt)
        job = response.stdout.strip().split(';')[0]
        if response.returncode or not job.isdigit():
            raise RuntimeError('Unconfirmed replacement: reconcile journal and Slurm, never retry blindly')
        receipt.update(confirmed_job_id=job, confirmed_at=time.time())
        ctl.write(replacement, receipt)
        ctl.write(jobs_path, dict(round=0, version=ctl.version(0), branches=[0,1,2,3], prepare=job,
                                 followon_dependency_recovery=str(replacement.relative_to(ctl.EXP))))
    return ctl.dispatch()


if __name__ == '__main__':
    print(json.dumps(recover(), indent=2))
