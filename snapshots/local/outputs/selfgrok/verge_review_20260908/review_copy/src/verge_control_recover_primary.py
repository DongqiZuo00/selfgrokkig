"""One bounded recovery of an expired, successful primary -> direct dependency.

No scientific configuration is changed. Preserve the failed submission intent,
check both Slurm stores, and journal the replacement before issuing sbatch.
An uncertain replacement must be reconciled manually, never submitted twice.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import verge_control_controller as ctl


def released_arguments(original, expected, dependency, accounting, live_returncode,
                       live_stderr, queue, matching_history):
    if original != expected or not dependency.isdigit():
        raise RuntimeError("Unexpected submission intent or dependency")
    rows = [line.strip().split('|') for line in accounting.splitlines() if line.strip()]
    parents = [row for row in rows if row[0] == dependency]
    if parents != [[dependency, 'COMPLETED', '0:0']]:
        raise RuntimeError("Exact primary job is not certified completed successfully")
    if live_returncode == 0 or 'Invalid job id specified' not in live_stderr:
        raise RuntimeError("Dependency is not confirmed purged from the controller")
    # This one-time handoff is only applicable before any new experiment wave.
    # Other work is left untouched; an unexpected job merely blocks recovery.
    if any(line.split('|')[-1].strip() != 'duj-github-snapshot'
           for line in queue.splitlines() if line.strip()):
        raise RuntimeError("A live job requires ownership/resource reconciliation")
    if matching_history.strip():
        raise RuntimeError("A previous control submission exists; do not duplicate it")
    needle = '--dependency=afterok:' + dependency
    if original.count(needle) != 1:
        raise RuntimeError("Ambiguous dependency")
    return [arg for arg in original if arg != needle]


def command(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, **kwargs)


def recover():
    if (os.environ.get('VERGE_BOOK_SUITE') != 'verge_book_v2'
            or os.environ.get('VERGE_CONTROL_SUITE') != 'verge_control_v1'):
        raise RuntimeError('Explicit suite environment required')
    with ctl.dispatch_lock():
        dependency = ctl.primary_finished()
        from verge_book_controller import schedule, verified_round
        for arm, r in schedule():
            verified_round(arm, r)
        for label in ctl.LABELS:
            cfg = ctl.read(ctl.EXP / 'manifests' / (ctl.version(label, 0) + '.json'))
            ctl.validate_launch(ctl.EXP, cfg)
        jobs_path = ctl.ROOT / 'jobs/round_00.json'
        replacement = ctl.ROOT / 'submission_intents/round_00_workers_expired_primary_recovery.json'
        if jobs_path.exists() or replacement.exists():
            raise RuntimeError('Handoff already attempted; inspect recorded jobs, never resubmit')
        intent_path = ctl.ROOT / 'submission_intents/round_00_workers.json'
        original = ctl.read(intent_path)
        if original.get('confirmed_job_id') is not None:
            raise RuntimeError('Original job already confirmed')
        expected = ['sbatch', '--parsable', '--dependency=afterok:' + dependency,
                    '--kill-on-invalid-dep=yes',
                    '--export=ALL,VERGE_BOOK_SUITE=verge_book_v2,VERGE_CONTROL_SUITE=verge_control_v1',
                    '--job-name=vc-r0-workers', '--array=0-2%2',
                    str(ctl.EXP / 'scripts/verge_control_workers.sbatch'), '0']
        account = command(['sacct', '-j', dependency, '--format=JobIDRaw,State,ExitCode', '-n', '-P'], check=True)
        live = command(['scontrol', 'show', 'job', dependency])
        queue = command(['squeue', '-h', '-u', 'jinjiaguo', '-o', '%i|%j'], check=True)
        history = command(['sacct', '-u', 'jinjiaguo', '-S', '2026-09-07T00:00:00',
                           '--name=vc-r0-workers', '--format=JobIDRaw,State,ExitCode', '-n', '-P'], check=True)
        args = released_arguments(original['arguments'], expected, dependency, account.stdout,
                                  live.returncode, live.stderr, queue.stdout, history.stdout)
        test = command([args[0], '--test-only', *args[1:]], cwd=ctl.EXP.parents[1])
        if test.returncode:
            raise RuntimeError('Replacement test-only rejected: ' + test.stderr)
        receipt = {'observed_at': time.time(), 'reason': 'Successful primary job purged past MinJobAge',
                   'primary_dependency': dependency, 'accounting': account.stdout,
                   'live_returncode': live.returncode, 'live_stderr': live.stderr,
                   'queue': queue.stdout, 'matching_control_history': history.stdout,
                   'original_intent': str(intent_path.relative_to(ctl.EXP)),
                   'original_intent_preserved': True, 'arguments': args,
                   'confirmed_job_id': None, 'scientific_protocol_changed': False}
        ctl.write(replacement, receipt)
        response = command(args, cwd=ctl.EXP.parents[1])
        receipt.update(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr)
        ctl.write(replacement, receipt)
        job = response.stdout.strip().split(';')[0]
        if response.returncode or not job.isdigit():
            raise RuntimeError('Unconfirmed recovery submission: inspect journal and Slurm before retry')
        receipt.update(confirmed_job_id=job, confirmed_at=time.time())
        ctl.write(replacement, receipt)
        ctl.write(jobs_path, {'round': 0, 'labels': list(ctl.LABELS), 'workers': job,
                             'primary_dependency_recovery': str(replacement.relative_to(ctl.EXP))})
    # The new worker job is still retained by Slurm; ordinary afterok applies.
    return ctl.dispatch()


if __name__ == '__main__':
    print(json.dumps(recover(), indent=2))
