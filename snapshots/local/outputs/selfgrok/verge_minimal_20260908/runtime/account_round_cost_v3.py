"""Three-round accounting with explicit zero-generation baseline reuse.

Extends the untouched v2 accounting tool; does not edit prior reports, runtime
contracts, backends, executors, scorers, checkpoints, or any scheduler job.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess

import account_round_cost_v2 as v2

PREPARE_JOB = '41419999'
PREPARE_DIRECTORY = 'round_prepare_ignition_41419999'
SOURCE_PREPARATION = 'round_prepare_named_41417194'


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_baseline_reuse(root, preparation, *, source_preparation=SOURCE_PREPARATION):
    """Verify actual original IDs/content, not merely a claimed cache flag."""
    root = Path(root).resolve()
    new = root / 'runs' / preparation
    old = root / 'runs' / source_preparation
    report = v2.read(new / 'baseline_reuse.json')
    copied = v2.read(new / 'base/fragment.json')
    original = v2.read(old / 'base/fragment.json')
    require(v2.source_path(report['source_preparation'], root) == old.resolve(), 'reuse names a different original preparation')
    for key, name in (('source_fragment', 'base/fragment.json'), ('source_plan', 'plan.json'), ('source_ready', 'READY.json')):
        actual = v2.source_path(report[key]['path'], root)
        require(actual == (old / name).resolve(), 'reuse provenance path changed: ' + key)
        require(v2.digest(actual.read_bytes()) == report[key]['file_sha256'], 'reuse provenance SHA changed: ' + key)
    require(report['evaluation_cache_hit'] is True and report['records_copied_without_any_change'] is True,
            'reuse must explicitly identify unchanged original records')
    require(report['reused_rollout_slots'] == 256 and report['new_evaluation_rollouts'] == 0
            and report['new_evaluation_generated_tokens'] == 0, 'reuse must record 256 old slots and zero new generation')
    require(copied['baseline_reuse'] == report, 'fragment/reuse report differ')
    require(copied['evaluations'] == original['evaluations'] and copied['checkpoints'] == original['checkpoints'],
            'copied baseline records or checkpoint identity changed')
    require(copied['baseline_contract_sha256'] == original['baseline_contract_sha256'] == report['baseline_contract_sha256'],
            'baseline contract changed across reuse')
    require(copied['original_execution'] == original['execution'] and original['execution']['no_training'] is True,
            'original baseline execution metadata changed or baseline involved training')
    require(copied['execution']['actual_generation_tokens'] == 0 and copied['execution']['new_evaluation_rollouts'] == 0
            and copied['execution']['reused_rollout_slots'] == 256, 'copied execution incorrectly charges old work again')
    require(not v2.raw_candidates(new / 'base/raw_evaluation', '*.json'),
            'new baseline raw files exist: reconcile their actual generation provenance rather than assume zero')
    sources = {}
    for entry in report['raw_sources']:
        path = v2.source_path(entry['path'], root)
        require(path.is_relative_to((old / 'base/raw_evaluation').resolve()), 'reused raw source is outside the original baseline')
        require(str(path) not in sources, 'duplicate source entry in baseline reuse list')
        data = path.read_bytes()
        require(v2.digest(data) == entry['file_sha256'], 'reused raw generation SHA changed')
        raw = json.loads(data)
        count = v2.native_counts(raw)
        require(not count['issues'] and count['rollouts'] == 8, 'reused source is not an intact eight-rollout native record')
        sources[str(path)] = {'sha256': entry['file_sha256'], 'raw': raw, 'tokens': count['native_generated_tokens']}
    require(len(sources) == 32, 'reuse must preserve exactly 32 original native generation chunks')
    records = copied['evaluations']['base']['records']
    require(len(records) == 256 and len({(r['instance_id'], r['sample_id']) for r in records}) == 256,
            'reuse must preserve 256 distinct original instance/sample slots')
    referenced = set()
    for record in records:
        filename, slot = record['generation_evidence_id'].rsplit('#sample', 1)
        path, slot = str(v2.source_path(filename, root)), int(slot)
        require(path in sources and 0 <= slot < 8, 'invalid original native generation slot')
        key = path, slot
        require(key not in referenced, 'same original generation slot represented twice')
        referenced.add(key)
        item = sources[path]
        require(record['raw_generation_sha256'] == item['sha256'], 'copied record has a different original SHA')
        require(record['completion_token_ids'] == item['raw']['completion_token_ids'][slot]
                and record['raw_completion'] == item['raw']['verifier_completions'][slot],
                'copied record differs from actual original token IDs or completion bytes')
    require(len(referenced) == 32 * 8, 'reuse omits an original generation slot')
    original_tokens = sum(item['tokens'] for item in sources.values())
    require(original_tokens == report['original_evaluation_generated_tokens'] == original['execution']['actual_generation_tokens'],
            'original baseline native cost differs from its retained counter')
    return {'status': 'verified', 'source_preparation': str(old.relative_to(root)),
            'new_preparation': str(new.relative_to(root)), 'original_job_charged': '41417194',
            'original_generated_tokens': original_tokens, 'new_baseline_generated_tokens': 0,
            'new_baseline_rollouts': 0, 'reused_rollout_slots': 256, 'reused_raw_sources': 32,
            'all_original_tokens_and_completion_bytes_match': True,
            'source_fragment_sha256': report['source_fragment']['file_sha256'],
            'baseline_reuse_file_sha256': v2.digest((new / 'baseline_reuse.json').read_bytes()),
            'sources': [{'path': str(Path(path).relative_to(root)), 'sha256': item['sha256'], 'native_tokens': item['tokens']}
                        for path, item in sources.items()]}


def three_round_specs(array_id=None):
    result = v2.specs() + [dict(job=PREPARE_JOB, label='round3_preparation', kind='prepare', directory=PREPARE_DIRECTORY)]
    if array_id is not None:
        require(array_id.isdigit(), 'provide the explicitly assigned Slurm round3 array ID')
        result.extend(dict(job=f'{array_id}_{index}', label=f'round3_{branch}', kind='branch', branch=branch,
                           directory=f'round3_{branch}_{array_id}_{index}', expected_budget=65536)
                      for index, branch in enumerate(v2.BRANCHES))
    return result


def audit(root, sacct_text, array_id=None):
    root = Path(root).resolve()
    result = v2.audit(root, sacct_text, three_round_specs(array_id))
    result['mode'] = 'three_round_actual_cost_with_verified_cross_round_baseline_reuse'
    result['round3_array_id'] = array_id
    result['round3_array_identity_known'] = array_id is not None
    result['validation_scope'] = {'cost_tool_tests_are_training_path_tests': False,
                                  'training_path_test_count_claimed_by_this_report': None}
    try:
        reuse = verify_baseline_reuse(root, PREPARE_DIRECTORY)
        charged = {item['source_path']: item for item in result['source_files']}
        for source in reuse['sources']:
            entry = charged.get(source['path'])
            require(entry and entry['job_id'] == '41417194' and entry['sha256'] == source['sha256']
                    and entry['native_generated_tokens'] == source['native_tokens'],
                    'reused baseline source is not charged exactly at its original known job')
        result['round3_baseline_reuse'] = reuse
    except FileNotFoundError as error:
        result['round3_baseline_reuse'] = {'status': 'pending', 'missing_path': str(error.filename)}
    except (ValueError, KeyError) as error:
        result['round3_baseline_reuse'] = {'status': 'needs_reconciliation', 'error': str(error)}
        result['issues'].append('round3 baseline reuse: ' + str(error))
    if array_id is None or result['round3_baseline_reuse']['status'] != 'verified' or result['issues']:
        result['status'] = 'snapshot_or_needs_reconciliation'
    return result


def markdown(result):
    totals, scheduler = result['generation_totals'], result['scheduler']
    lines = ['# VERGE 三轮累计成本与预算', '', f"状态：**{result['status']}**；UTC 统计时间：{result['created_utc']}。",
             '', f"累计保存的实际 native generation：**{result['actual_saved_native_generated_tokens']:,} tokens**。Slurm 分配：**{scheduler['allocated_b200_hours']:.6f} B200 小时**。原两轮 `COST_AND_BUDGET.md/json` 保留。",
             '', '| 类别 | 实际 native tokens |', '|---|---:|']
    lines.extend(f'| {key} | {totals[key]:,} |' for key in v2.CATEGORIES)
    lines += ['', '| 作业/阶段 | Solver train | Solver eval 新生成 | Q | Stage probe |', '|---|---:|---:|---:|---:|']
    lines.extend('| ' + row['label'] + ' | ' + ' | '.join(f"{row['generation'][key]:,}" for key in v2.CATEGORIES) + ' |'
                 for row in result['runs'])
    reuse = result['round3_baseline_reuse']
    lines += ['', '## 第三轮 baseline 复用', '']
    if reuse['status'] == 'verified':
        lines += [f"新增 baseline 为 **0 tokens / 0 rollouts**。256 个 slots、32 个原始 generation chunks 的 token IDs、completion 内容及 SHA 全部与 41417194 保持一致。原先 **{reuse['original_generated_tokens']:,} tokens** 只在第二轮 preparation 计一次。第三轮复制的 fragment、original_execution 和逻辑评估别名均未另计 generation。"]
    else:
        lines += [f"复用核验尚未完成：{reuse}。不将缺失证据当成已确认的零成本。"]
    lines += ['', '## Slurm 分配', '', '| Allocation | State | Seconds | B200 小时 |', '|---|---|---:|---:|']
    lines.extend(f"| {row['job_id']} | {row['state']} | {row['elapsed_seconds']} | {row['allocated_b200_seconds']/3600:.6f} |"
                 for row in scheduler['allocations'])
    lines += ['', f"已列入作业的峰值同时分配为 **{scheduler['peak_named_job_allocated_gpus']} GPU / {scheduler['peak_named_job_allocated_memory_gib']:g} GiB**；用户上限为 4 GPU / 192 GiB。这是 allocation，不能解读为实际 GPU 利用率、进程峰值内存或所有其他作业的总资源。",
              '', '失败分配照录，包括 41410512。其 ElapsedRaw=0 是 sacct 的秒级记录，未虚构亚秒耗时；没有保存的 raw 不被误称为测得模型 generation=0。',
              '', '## 计数与验证范围', '',
              '逐个实际 generation 来源核算 native token IDs，包含 EOS，排除 EOS 后 batch padding、prompt 和固定格式前缀。独立采样即使输出相同仍分别计数；缓存 alias 或复制 fragment 只引用原来源。Stage proof 复用 545/822 个 screen tokens，不产生额外 generation；更新计算已包含在其作业的分配时长内。',
              '', '第三轮沿用第二轮 Solver prompt view / grammar / 原始基线，并增加已声明的 Challenger 第一阶段 proposal prior。这里仅比较成本，不把不同 prompt view 或 proposal policy 的三轮合并成同一效果实验。此前 release 的 smokes、CPU-only 语法检查、其他项目和未明确列入的作业不在 GPU 作业统计中。',
              '', f"核验 {result['unique_generation_source_files']} 个独立 generation 来源；missing scheduler tasks={scheduler['missing_expected_jobs']}；全部指定任务终止={scheduler['all_named_jobs_terminal']}。"]
    validation = result.get('cpu_validation', {})
    lines += ['', f"本报告的 CPU 日志 `{validation.get('log', '尚未关联')}`：{validation.get('tests', '未记录')} 项，完成标记={validation.get('passed', '未记录')}。这些全部是**成本/来源审计工具测试**，包括此前 v2 的 12 项；不并入 Solver、Challenger 或训练路径测试数量。训练路径测试由相应执行报告单独统计。", '', '审计问题：']
    lines += [''] + (['- ' + issue for issue in result['issues']] if result['issues'] else ['无已发现的计数或来源问题；最终状态还要求第三轮 array 身份已知、全部任务终止且 baseline 复用核验通过。'])
    lines += ['', '复现使用新文件名，保留已有报告：', '', '```bash',
              "'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B -m unittest discover -s runtime -p 'test_account_round_cost_v*.py' -v",
              f"'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B runtime/account_round_cost_v3.py --round3-array {result['round3_array_id'] or 'EXPLICIT_ARRAY_ID'} --output evidence/COST_THREE_ROUNDS_RECHECK_NEW.json --markdown evidence/COST_THREE_ROUNDS_RECHECK_NEW.md", '```']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=v2.ROOT)
    parser.add_argument('--round3-array')
    parser.add_argument('--sacct-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for path in (args.output, args.markdown):
        if path is not None:
            require(not path.exists() and path.resolve().is_relative_to(root), 'use new report paths inside this release')
    jobs = list(v2.NAMED_JOBS) + [PREPARE_JOB] + ([args.round3_array] if args.round3_array else [])
    command = ['sacct', '--array', '-X', '-j', ','.join(jobs), '-P', '-n', '--format=' + ','.join(v2.FIELDS)]
    text = args.sacct_file.read_text() if args.sacct_file else subprocess.run(command, text=True, capture_output=True, check=True).stdout
    result = audit(root, text, args.round3_array)
    result.update(scheduler_command=command, scheduler_raw=text, auditor_source_sha256=v2.digest(Path(__file__).read_bytes()),
                  retained_v2_auditor_sha256=v2.digest(Path(v2.__file__).read_bytes()))
    old_report = root / 'evidence/COST_AND_BUDGET.json'
    if old_report.exists():
        result['retained_two_round_report'] = {'path': str(old_report.relative_to(root)), 'sha256': v2.digest(old_report.read_bytes())}
    log = root / 'evidence/COST_V3_AUDIT_TOOL_CPU_TESTS.log'
    if log.exists():
        data = log.read_bytes()
        content = data.decode('utf-8', errors='replace')
        match = re.search(r'^Ran (\d+) tests?', content, re.MULTILINE)
        result['cpu_validation'] = {'log': str(log.relative_to(root)), 'sha256': v2.digest(data),
                                    'tests': int(match[1]) if match else None, 'passed': content.rstrip().endswith('OK'),
                                    'suite_role': 'cost accounting and provenance audit tools only', 'training_path_tests': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(result), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'actual_saved_native_generated_tokens', 'generation_totals', 'issues')}, indent=2))
    print(json.dumps({'reuse_status': result['round3_baseline_reuse']['status'],
                      'all_named_jobs_terminal': result['scheduler']['all_named_jobs_terminal'],
                      'allocated_b200_hours': result['scheduler']['allocated_b200_hours']}, indent=2))


if __name__ == '__main__':
    main()
