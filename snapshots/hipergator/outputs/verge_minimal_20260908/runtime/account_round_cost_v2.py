"""Independent, stdlib-only audit of named VERGE raw generation and allocation.

Counts canonical generation-source files, never copied fragment/sample aliases.
No model imports, scheduler mutations, training, or rewriting of existing runs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = '/blue/du.j/jinjiaguo/self grok/experiments/has_transfer_witness/outputs/verge_minimal_20260908'
NAMED_JOBS = ('41410512', '41410864', '41412533', '41413994', '41414619', '41415498', '41417194', '41418169')
BRANCHES = ('g1', 'g2', 'g3', 'direct')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT', 'LAUNCH_FAILED'}
FIELDS = ('JobID', 'JobIDRaw', 'State', 'ElapsedRaw', 'AllocTRES', 'ExitCode', 'Start', 'End', 'Submit', 'Partition')
CATEGORIES = ('solver_training', 'solver_evaluation', 'challenger', 'stage_probe')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def specs():
    result = [dict(job='41410512', label='failed_initial_stage_launch', kind='failed_launch', directory=None),
              dict(job='41410864', label='unconstrained_stage_screen', kind='stage', directory='stage_screen_retry1_41410864'),
              dict(job='41412533', label='grammar_stage_screen', kind='stage', directory='stage_grammar_v1_41412533'),
              dict(job='41415498', label='naming_v2_stage_screen', kind='stage', directory='naming_v2_41415498')]
    for number, prepare, array in ((1, 'round_prepare_41413994', '41414619'),
                                   (2, 'round_prepare_named_41417194', '41418169')):
        result.append(dict(job=prepare.rsplit('_', 1)[1], label=f'round{number}_preparation', kind='prepare', directory=prepare))
        result.extend(dict(job=f'{array}_{index}', label=f'round{number}_{branch}', kind='branch',
                           branch=branch, directory=f'round{number}_{branch}_{array}_{index}', expected_budget=65536)
                      for index, branch in enumerate(BRANCHES))
    return result


def source_path(value, root):
    """Map only the known remote release or a path genuinely inside this root."""
    text = str(value).replace('\\', '/').split('#sample', 1)[0]
    candidate = Path(text)
    if candidate.is_absolute() and candidate.resolve().is_relative_to(root.resolve()):
        path = candidate
    elif text.startswith(REMOTE_ROOT + '/'):
        path = root / text[len(REMOTE_ROOT) + 1:]
    else:
        path = Path(text)
        if not path.is_absolute():
            path = root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('generation source escapes the named selfgrok release')
    return path


def native_counts(record, eos_ids=(2,)):
    """Actual suffix IDs through first EOS; report rather than hide bad records."""
    values = record['completion_token_ids']
    if not isinstance(values, list) or not values:
        raise ValueError('nonempty completion_token_ids required')
    rows = values if isinstance(values[0], list) else [values]
    if any(not isinstance(row, list) or not row or any(type(t) is not int or t < 0 for t in row) for row in rows):
        raise ValueError('native suffixes must be nonempty lists of nonnegative integer token IDs')
    errors, lengths, raw_lengths = [], [], list(map(len, rows))
    for row in rows:
        first = next((index + 1 for index, token in enumerate(row) if token in eos_ids), len(row))
        lengths.append(first)
    if raw_lengths != lengths:
        errors.append('saved suffix contains post-EOS IDs; native count excludes them and audit is not clean')
    total = sum(lengths)
    if record.get('generated_tokens') != total:
        errors.append('reported generated_tokens differs from actual native IDs')
    if record.get('n', len(rows)) != len(rows):
        errors.append('reported sample count differs from native rows')
    cap = record.get('max_new_tokens')
    if cap is not None and (type(cap) is not int or cap < 1 or any(length > cap for length in lengths)):
        errors.append('native suffix exceeds its stated cap')
    return {'native_generated_tokens': total, 'raw_saved_id_count': sum(raw_lengths), 'rollouts': len(rows),
            'suffix_lengths': lengths, 'reported_generated_tokens': record.get('generated_tokens'), 'issues': errors}


def raw_candidates(directory, pattern):
    """An atomic rewrite is the same generation, not another source file."""
    candidates = {p.resolve(): p for p in directory.glob(pattern)}
    for temporary in directory.glob(pattern + '.writing'):
        final = Path(str(temporary)[:-len('.writing')]).resolve()
        candidates.setdefault(final, temporary)
    return candidates


def parse_scheduler(text, expected_jobs):
    rows, ignored, issues = {}, [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.rstrip('|').split('|')
        if len(values) != len(FIELDS):
            issues.append('malformed sacct row: ' + line)
            continue
        raw = dict(zip(FIELDS, values))
        job = raw['JobID']
        if '.' in job or '[' in job or job not in expected_jobs:
            ignored.append(job)
            continue
        if job in rows:
            issues.append('duplicate allocation row needs explicit reconciliation: ' + job)
            continue
        tres = dict(item.split('=', 1) for item in raw['AllocTRES'].split(',') if '=' in item)
        typed = {key: int(value) for key, value in tres.items() if key.startswith('gres/gpu:')}
        gpus = int(tres['gres/gpu']) if 'gres/gpu' in tres else sum(typed.values())
        b200 = int(tres.get('gres/gpu:b200', gpus if raw['Partition'] == 'hpg-b200' else 0))
        seconds = int(raw['ElapsedRaw'] or 0)
        memory = tres.get('mem', '0')
        scale = {'K': 1 / 1024**2, 'M': 1 / 1024, 'G': 1, 'T': 1024}
        memory_gib = float(memory[:-1]) * scale[memory[-1]] if memory[-1:] in scale else float(memory) / 1024
        state = raw['State'].split()[0].rstrip('+')
        rows[job] = {'job_id': job, 'job_id_raw': raw['JobIDRaw'], 'state': state, 'exit_code': raw['ExitCode'],
                     'elapsed_seconds': seconds, 'allocated_gpus': gpus, 'allocated_b200': b200,
                     'allocated_memory_gib': memory_gib, 'allocated_tres': tres,
                     'allocated_gpu_seconds': seconds * gpus, 'allocated_b200_seconds': seconds * b200,
                     'start': raw['Start'], 'end': raw['End'], 'submit': raw['Submit'], 'partition': raw['Partition']}
    missing = sorted(set(expected_jobs) - set(rows))
    peak_gpu = peak_mem = active_gpu = active_mem = 0
    events = []
    for row in rows.values():
        if row['start'] in ('Unknown', 'None', ''):
            continue
        start = datetime.fromisoformat(row['start'])
        end = datetime.fromisoformat(row['end']) if row['end'] not in ('Unknown', 'None', '') else datetime.now()
        if end <= start:
            continue
        events.extend(((start, 1, row['allocated_gpus'], row['allocated_memory_gib']),
                       (end, -1, -row['allocated_gpus'], -row['allocated_memory_gib'])))
    for _, _, gpu, mem in sorted(events):  # Release at a boundary precedes a new allocation.
        active_gpu += gpu
        active_mem += mem
        peak_gpu, peak_mem = max(peak_gpu, active_gpu), max(peak_mem, active_mem)
    return {'allocations': list(rows.values()), 'missing_expected_jobs': missing, 'ignored_rows': ignored,
            'issues': issues, 'all_named_jobs_terminal': not missing and not issues and bool(rows)
                  and all(row['state'] in TERMINAL for row in rows.values()),
            'allocated_gpu_hours': sum(r['allocated_gpu_seconds'] for r in rows.values()) / 3600,
            'allocated_b200_hours': sum(r['allocated_b200_seconds'] for r in rows.values()) / 3600,
            'peak_named_job_allocated_gpus': peak_gpu, 'peak_named_job_allocated_memory_gib': peak_mem,
            'named_job_peak_within_user_ceiling': peak_gpu <= 4 and peak_mem <= 192,
            'elapsed_resolution': 'whole seconds reported by sacct; no invented sub-second duration',
            'allocation_is_not_measured_GPU_utilization': True}


def journal_sources(path, root):
    if not path.exists():
        return {}, {'present': False, 'issues': []}
    data = path.read_bytes()
    previous, sources, issues, updates, pending = '0' * 64, {}, [], 0, {}
    for index, line in enumerate(data.decode('utf-8').splitlines()):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            issues.append('incomplete/malformed journal line ' + str(index))
            break
        actual = item.pop('sha256')
        wanted = digest(json.dumps(item, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
        if actual != wanted or item['sequence'] != index or item['previous_sha256'] != previous:
            issues.append('journal hash chain mismatch at ' + str(index))
        previous = actual
        payload = item['payload']
        if item['kind'] == 'generation_attempt':
            pending = payload
        elif item['kind'] == 'generation_source':
            key = str(source_path(payload['raw_path'], root))
            if key in sources:
                issues.append('duplicate training generation provenance: ' + key)
            sources[key] = {**payload, 'boundary_drain': pending.get('drain')}
        elif item['kind'] == 'actual_model_update':
            updates += 1
    return sources, {'present': True, 'sha256': digest(data), 'issues': issues, 'actual_update_events': updates}


def audit(root, sacct_text, run_specs=None):
    root = Path(root).resolve()
    run_specs = specs() if run_specs is None else run_specs
    scheduler = parse_scheduler(sacct_text, {item['job'] for item in run_specs})
    states = {row['job_id']: row['state'] for row in scheduler['allocations']}
    sources, reports, issues, proofs, references = {}, [], [], [], {}
    totals = Counter({key: 0 for key in CATEGORIES})
    for spec in run_specs:
        report = {**spec, 'raw_files': 0, 'generation': {key: 0 for key in CATEGORIES}, 'rollouts': 0,
                  'scheduler_state': states.get(spec['job']), 'issues': []}
        reports.append(report)
        if spec['directory'] is None:
            report.update(saved_generation_known=0, actual_generation_completeness='no run directory/source; no assumption beyond saved evidence')
            continue
        directory = root / 'runs' / spec['directory']
        report['run_directory_exists'] = directory.is_dir()
        if spec['kind'] == 'stage':
            patterns = [('stage_probe', directory / 'samples', '*.json')]
        elif spec['kind'] == 'prepare':
            patterns = [('solver_evaluation', directory / 'base/raw_evaluation', '*.json'),
                        ('challenger', directory, 'challenger_sample_*.json')]
        else:
            patterns = [('solver_training', directory / 'execution/raw_training', '*.json'),
                        ('solver_evaluation', directory / 'execution/raw_evaluation', '*.json')]
        journal, journal_report = journal_sources(directory / 'execution/training_journal.jsonl', root)
        if spec['kind'] == 'branch':
            report['journal'] = journal_report
            report['issues'].extend(journal_report['issues'])
            report['training_by_task_kind'] = dict(target=0, stage=0, unknown=0)
            report['boundary_drain_tokens'] = 0
        for category, folder, pattern in patterns:
            for origin, actual in sorted(raw_candidates(folder, pattern).items()):
                key = str(origin)
                if key in sources:
                    report['issues'].append('same canonical generation source assigned twice: ' + key)
                    continue
                try:
                    data = actual.read_bytes()
                    raw = json.loads(data)
                    item = native_counts(raw)
                except (ValueError, KeyError, OSError) as error:
                    report['issues'].append(f'unreadable/incomplete raw source {actual}: {error}')
                    continue
                item.update(source_path=str(origin.relative_to(root)), actual_file=str(actual.relative_to(root)),
                            sha256=digest(data), category=category, job_id=spec['job'], label=spec['label'],
                            from_atomic_temporary=actual != origin, seed=raw.get('seed'), instance_id=raw.get('instance_id'),
                            decoding_policy=raw.get('decoding_policy'), optimizer_steps_before=raw.get('optimizer_steps_before'))
                sources[key] = item
                count = item['native_generated_tokens']
                report['raw_files'] += 1
                report['rollouts'] += item['rollouts']
                report['generation'][category] += count
                totals[category] += count
                report['issues'].extend(f'{origin.name}: {issue}' for issue in item['issues'])
                if category == 'solver_training':
                    provenance = journal.get(key)
                    kind = 'target' if provenance and provenance['target'] else 'stage' if provenance else 'unknown'
                    report['training_by_task_kind'][kind] += count
                    if provenance:
                        if provenance['raw_sha256'] != item['sha256']:
                            report['issues'].append('training source digest mismatch: ' + key)
                        report['boundary_drain_tokens'] += count if provenance['boundary_drain'] else 0
                    else:
                        report['issues'].append('saved training generation lacks a journal source: ' + key)
        fragment_path = directory / ('base/fragment.json' if spec['kind'] == 'prepare' else 'fragment.json')
        if fragment_path.exists():
            fragment = read(fragment_path)
            evaluations = fragment.get('evaluations', {})
            report['evaluation_aliases'] = len(evaluations)
            report['fully_cached_evaluation_aliases'] = sum(bool(e.get('evaluation_cache_hit')) for e in evaluations.values())
            for evaluation in evaluations.values():
                for record in evaluation.get('records', []):
                    if 'generation_evidence_id' in record:
                        original = str(source_path(record['generation_evidence_id'], root))
                        sha = record['raw_generation_sha256']
                        if original in references and references[original] != sha:
                            report['issues'].append('same evaluation generation origin has conflicting content SHA: ' + original)
                        references[original] = sha
            if spec['kind'] == 'prepare':
                report['baseline_counter_matches_native_sources'] = fragment['execution']['actual_generation_tokens'] == report['generation']['solver_evaluation']
                if not report['baseline_counter_matches_native_sources']:
                    report['issues'].append('baseline counter differs from its new source-file native count')
            if spec['kind'] == 'branch':
                complete = (directory / 'COMPLETE.json').exists()
                report['complete_marker_present'] = complete
                if complete:
                    execution = fragment['execution']
                    ledgers = fragment['branches'][spec['branch']]['phases']
                    total = report['generation']['solver_training']
                    report['budget_check_passed'] = total == spec['expected_budget'] == execution['training_generated_tokens'] == sum(p['generated_tokens'] for p in ledgers)
                    report['actual_optimizer_steps'] = execution['actual_optimizer_steps']
                    report['nonzero_advantage_tokens'] = sum(p['nonzero_advantage_tokens'] for p in ledgers)
                    report['backend_generation_counter_matches'] = execution['backend_tokens_all_generation_calls'] == total + report['generation']['solver_evaluation']
                    if not report['budget_check_passed'] or not report['backend_generation_counter_matches']:
                        report['issues'].append('completed branch native source count differs from B/ledger/backend')
        proof_path = directory / 'one_binary_update_proof.json'
        if proof_path.exists():
            proof = read(proof_path)
            original = str(source_path(proof['selected_stage']['raw_path'], root))
            source = sources.get(original)
            reused = proof['update']['tokens_used']
            valid = source is not None and source['native_generated_tokens'] == reused
            proofs.append({'job_id': spec['job'], 'proof_file': str(proof_path.relative_to(root)),
                           'source_raw': str(Path(original).relative_to(root)), 'reused_update_input_tokens': reused,
                           'additional_generated_tokens': 0, 'source_count_verified': valid,
                           'optimizer_step': proof['update']['optimizer_step']})
            if not valid:
                report['issues'].append('proof does not match the already-counted screen generation')
        if spec['kind'] == 'stage' and (directory / 'SUMMARY.json').exists():
            summary = read(directory / 'SUMMARY.json')
            groups = summary.get('summaries', [])
            report['screen_summary_matches_raw'] = len(groups) == report['raw_files'] and sum(g['generated_tokens'] for g in groups) == report['generation']['stage_probe']
            if summary.get('status') == 'completed' and not report['screen_summary_matches_raw']:
                report['issues'].append('completed stage summary differs from its raw generation sources')
        if states.get(spec['job']) == 'COMPLETED':
            if not directory.is_dir() or report['raw_files'] == 0:
                report['issues'].append('completed named job has no saved generation sources')
            if spec['kind'] == 'branch' and not report.get('complete_marker_present'):
                report['issues'].append('scheduler completed branch has no complete evidence marker')
        issues.extend(f"{spec['label']}: {issue}" for issue in report['issues'])
    for path, expected in references.items():
        if path in sources and sources[path]['sha256'] != expected:
            issues.append('evaluation reference/source content mismatch: ' + path)
        elif path not in sources:
            issues.append('evaluation references an uncounted origin; explicitly classify it before a final total: ' + path)
    issues.extend(scheduler['issues'])
    return {'mode': 'actual_native_generation_source_and_scheduler_allocation_audit_v2',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'status': 'complete' if scheduler['all_named_jobs_terminal'] and not issues else 'snapshot_or_needs_reconciliation',
            'generation_totals': dict(totals), 'actual_saved_native_generated_tokens': sum(totals.values()),
            'runs': reports, 'source_files': list(sources.values()), 'unique_generation_source_files': len(sources),
            'evaluation_reference_origins_checked': len(references), 'proof_updates': proofs,
            'scheduler': scheduler, 'issues': issues,
            'accounting_rules': {'EOS_included': True, 'post_EOS_batch_pad_excluded': True,
                'fixed_format_prefix_and_prompt_excluded_from_generated_budget': True,
                'cached_fragment_and_evaluation_aliases_add_no_generation': True,
                'same_token_content_in_independent_generation_files_is_not_deduplicated': True,
                'proof_update_input_is_reused_not_new_generation': True,
                'failed_allocations_are_included': True, 'CPU_or_GPU_utilization_not_inferred': True,
                'scope_is_explicit_named_jobs_only': True}}


def markdown(result):
    t, s = result['generation_totals'], result['scheduler']
    lines = ['# VERGE 实际成本与预算', '', f"状态：**{result['status']}**；统计时间：{result['created_utc']}。",
             '', f"实际保存的 native generation 共 **{result['actual_saved_native_generated_tokens']:,} tokens**；Slurm 分配 **{s['allocated_b200_hours']:.6f} B200 小时**。",
             '', '| 类别 | 实际 native tokens |', '|---|---:|']
    lines.extend(f'| {key} | {t[key]:,} |' for key in CATEGORIES)
    lines += ['', '| 作业/阶段 | Solver train | Solver eval 新生成 | Q | Stage probe |', '|---|---:|---:|---:|---:|']
    for row in result['runs']:
        values = row['generation']
        lines.append('| ' + row['label'] + ' | ' + ' | '.join(f'{values[k]:,}' for k in CATEGORIES) + ' |')
    lines += ['', '| Slurm allocation | State | Elapsed seconds | B200 小时 |', '|---|---|---:|---:|']
    lines.extend(f"| {r['job_id']} | {r['state']} | {r['elapsed_seconds']} | {r['allocated_b200_seconds']/3600:.6f} |" for r in s['allocations'])
    lines += ['', f"这些已命名作业的最大同时分配为 {s['peak_named_job_allocated_gpus']} GPU、{s['peak_named_job_allocated_memory_gib']:g} GiB；与用户上限 4 GPU / 192 GiB 的比较为 {s['named_job_peak_within_user_ceiling']}。这不包含其他项目或未列入的作业，也不是实际 GPU 利用率或进程峰值内存。",
              '', '训练预算以真实 raw token IDs 对照 phase ledger；通过 EOS 的最后一个 token计入，固定格式前缀、prompt 和 EOS 后 batch padding 不计入。评估仅计算实际原始生成文件，缓存 alias/复制的 fragment 不增加 generation。相同 token 内容来自两次独立生成仍算两次。',
              '', 'Stage proof 复用已计入 screen 的输入 tokens；其 optimizer 计算已经包含在该 Slurm 作业的分配时间内，不再加一份 generation：']
    lines.extend(f"- {p['job_id']}: 复用 {p['reused_update_input_tokens']:,} tokens，新增 generation 0，来源核验 {p['source_count_verified']}。" for p in result['proof_updates'])
    lines += ['', '失败作业照录，不删去其耗时。41410512 的 sacct ElapsedRaw 为 0 秒时，只能报告该秒级记录为 0，不能据此推断未被计量的亚秒开销；没有保存的 raw 也不被伪称为已测得模型生成 0。',
              '', '第一、二轮使用不同 prompt view，其成本分别列出；该报告不把两轮结果当成同一个固定 policy 实验。此前 release 的 GPU smokes、CPU 语法检查以及其他项目不在本次八个主 job ID 的统计范围内。',
              '', '## 完整性检查', '', f"唯一 generation 来源文件：{result['unique_generation_source_files']}；核对 evaluation 原始来源：{result['evaluation_reference_origins_checked']}；缺失预期 scheduler tasks：{s['missing_expected_jobs']}。"]
    lines.extend('- ' + issue for issue in result['issues'])
    if not result['issues']:
        lines.append('全部已执行的来源、预算、计数和 proof 复用核查通过。最终状态另要求每个预期 Slurm task 都已终止。')
    validation = result.get('cpu_validation', {})
    lines += ['', '## 审计工具与 CPU 日志', '',
              '新工具为 `runtime/account_round_cost_v2.py`，只使用 Python 标准库；原 `account_round_cost.py` 及冻结的训练/执行/评分代码不变。旧工具的来源目录计数方式保留；新版补齐第二轮、完整 array task 清单、缺失任务检查、原始来源 SHA 与训练 ledger 交叉核验、复用 proof 核验和分配峰值。',
              '', f"CPU 回归：{validation.get('tests', '未记录')} 项，日志通过标记 {validation.get('passed', '未记录')}；见 `{validation.get('log', '未记录')}`。首次远端 CPU 测试日志 `COST_V2_CPU_TESTS.log` 保留，其中嵌套 fixture 绝对路径被重复映射的问题已修复；它不涉及 GPU 训练代码。",
              '', '对应 JSON 保留每个原始 generation 文件的路径、内容 SHA、native 长度、分类以及原始 sacct 输出，能够复查本表。复现时请使用新的输出文件名，工具拒绝覆盖已有报告。',
              '', '```bash',
              "'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B -m unittest discover -s runtime -p test_account_round_cost_v2.py -v",
              "'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/account_round_cost_v2.py --output evidence/COST_RECHECK_NEW.json --markdown evidence/COST_RECHECK_NEW.md",
              '```']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--sacct-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for output in (args.output, args.markdown):
        if output is not None and (output.exists() or not output.resolve().is_relative_to(root)):
            raise ValueError('choose new audit outputs inside this selfgrok release; preserve previous snapshots')
    command = ['sacct', '--array', '-X', '-j', ','.join(NAMED_JOBS), '-P', '-n', '--format=' + ','.join(FIELDS)]
    scheduler_text = args.sacct_file.read_text() if args.sacct_file else subprocess.run(command, text=True, check=True, capture_output=True).stdout
    result = audit(root, scheduler_text)
    result['scheduler_command'] = command
    result['scheduler_raw'] = scheduler_text
    result['auditor_source_sha256'] = digest(Path(__file__).read_bytes())
    validation_log = root / 'evidence/COST_V2_CPU_TESTS_2.log'
    if validation_log.exists():
        contents = validation_log.read_bytes()
        text = contents.decode('utf-8', errors='replace')
        match = re.search(r'^Ran (\d+) tests?', text, re.MULTILINE)
        result['cpu_validation'] = {'log': str(validation_log.relative_to(root)), 'sha256': digest(contents),
                                    'tests': int(match[1]) if match else None, 'passed': text.rstrip().endswith('OK')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'actual_saved_native_generated_tokens', 'generation_totals', 'unique_generation_source_files', 'issues')}, indent=2))
    print(json.dumps({'all_named_jobs_terminal': result['scheduler']['all_named_jobs_terminal'],
                      'allocated_b200_hours': result['scheduler']['allocated_b200_hours']}, indent=2))


if __name__ == '__main__':
    main()
