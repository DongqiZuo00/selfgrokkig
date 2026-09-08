"""Read-only diagnosis of the twelve completed Solver branch journals."""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
import account_round_cost_v2 as cost


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def totals(groups):
    c = collections.Counter()
    for g in groups:
        c['generation_batches'] += 1
        c['native_tokens'] += g['native_tokens']
        c['rollouts_including_drains'] += len(g['rewards'])
        c['positive_rollouts_including_drains'] += sum(g['rewards'])
        if g['drain']:
            c['drain_batches'] += 1
            c['drain_tokens'] += g['native_tokens']
            c['drain_rollouts'] += len(g['rewards'])
        else:
            c['full_groups'] += 1
            c[g['reward_pattern'] + '_groups'] += 1
            c['full_group_rollouts'] += len(g['rewards'])
            c['positive_full_group_rollouts'] += sum(g['rewards'])
            c['non_drain_native_tokens'] += g['native_tokens']
            if g['reward_pattern'] == 'mixed':
                c['nonzero_advantage_native_tokens'] += g['native_tokens']
    for k in ('all_zero_groups', 'all_one_groups', 'mixed_groups', 'drain_batches', 'drain_tokens',
              'drain_rollouts', 'positive_rollouts_including_drains', 'positive_full_group_rollouts',
              'nonzero_advantage_native_tokens'):
        c.setdefault(k, 0)
    result = dict(c)
    result['nonzero_advantage_native_fraction'] = c['nonzero_advantage_native_tokens'] / c['native_tokens'] if c['native_tokens'] else 0
    result['positive_full_group_rollout_fraction'] = c['positive_full_group_rollouts'] / c['full_group_rollouts'] if c['full_group_rollouts'] else 0
    return result


def main():
    audited = read(ROOT / 'evidence/COST_THREE_ROUNDS.json')
    assert audited['status'] == 'complete' and not audited['issues']
    branches, all_groups, positives = [], [], []
    for prior in audited['runs']:
        if prior['kind'] != 'branch':
            continue
        directory = ROOT / 'runs' / prior['directory']
        path = directory / 'execution/training_journal.jsonl'
        fragment = read(directory / 'fragment.json')
        branch = prior['branch']
        proof = fragment['branches'][branch]
        _, journal_check = cost.journal_sources(path, ROOT)
        assert not journal_check['issues']
        assert digest(path) == prior['journal']['sha256'] == proof['journal_sha256']
        journal = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        groups, updates, phase_ledgers = [], [], {}
        source = None
        for event in journal:
            payload = event['payload']
            if event['kind'] == 'generation_source':
                source = payload
            elif event['kind'] == 'rollouts':
                records = payload['records']
                assert source and source['phase'] == payload['phase']
                assert len({r['prompt_id'] for r in records}) == 1
                assert all(r['reward'] in (0, 1) for r in records)
                lengths = [len(r['completion_token_ids']) for r in records]
                assert all(2 not in r['completion_token_ids'][:-1] for r in records)
                rewards = [r['reward'] for r in records]
                pattern = 'all_zero' if not any(rewards) else 'all_one' if all(rewards) else 'mixed'
                g = {'run': prior['label'], 'phase': payload['phase'], 'journal_line': event['sequence'] + 1,
                     'prompt_id': records[0]['prompt_id'], 'task_kind': records[0]['task_kind'],
                     'drain': payload['drain'], 'cap': payload['cap'], 'rewards': rewards,
                     'native_lengths': lengths, 'native_tokens': sum(lengths), 'reward_pattern': pattern,
                     'raw_path': source['raw_path'], 'raw_sha256': source['raw_sha256'],
                     'optimizer_steps_before': source['optimizer_steps_before']}
                if not g['drain']:
                    assert len(records) == 8
                groups.append(g)
                for i, record in enumerate(records):
                    if record['reward']:
                        positives.append({**{k: g[k] for k in ('run', 'phase', 'prompt_id', 'task_kind', 'journal_line', 'raw_path', 'raw_sha256')},
                                          'sample': i, 'native_tokens': lengths[i], 'completion': record['raw_completion']})
            elif event['kind'] == 'actual_model_update':
                updates.append({'journal_line': event['sequence'] + 1, **payload})
            elif event['kind'] == 'phase_complete':
                phase_ledgers[payload['name']] = payload
        summary = totals(groups)
        assert summary['native_tokens'] == prior['generation']['solver_training'] == 65536
        assert len(updates) == summary['mixed_groups'] == fragment['execution']['actual_optimizer_steps']
        assert summary['nonzero_advantage_native_tokens'] == sum(p['nonzero_advantage_tokens'] for p in proof['phases'])
        phases = []
        for phase in proof['phases']:
            assert phase == phase_ledgers[phase['name']]
            selected = [g for g in groups if g['phase'] == phase['name']]
            t = totals(selected)
            assert t['native_tokens'] == phase['generated_tokens']
            assert t['nonzero_advantage_native_tokens'] == phase['nonzero_advantage_tokens']
            assert t['non_drain_native_tokens'] == phase['loss_eligible_tokens']
            assert t['all_zero_groups'] + t['all_one_groups'] == phase['constant_groups']
            phases.append({'phase': phase['name'], **t, 'ledger': phase,
                           'by_prompt': {p: totals([g for g in selected if g['prompt_id'] == p]) for p in sorted({g['prompt_id'] for g in selected})}})
        branches.append({'run': prior['label'], **summary, 'optimizer_updates': len(updates),
                         'journal_path': str(path.relative_to(ROOT)), 'journal_sha256': digest(path),
                         'base_checkpoint': proof['base_checkpoint'], 'fresh_optimizer': proof['fresh_optimizer'],
                         'optimizer_initial_state_entries': proof['optimizer_initial_state_entries'],
                         'initial_parameter_fingerprint': fragment['execution']['initial_parameter_fingerprint'],
                         'final_parameter_fingerprint': fragment['execution']['final_parameter_fingerprint'],
                         'phases': phases, 'updates': updates})
        all_groups.extend(groups)
    preparation_dirs = ('round_prepare_41413994', 'round_prepare_named_41417194', 'round_prepare_ignition_41419999')
    rounds = []
    for i, preparation in enumerate(preparation_dirs, 1):
        plan = read(ROOT / 'runs' / preparation / 'plan.json')
        archive = read(ROOT / 'runs' / f'round{i}_result/archive.json')
        buffer = read(ROOT / 'runs' / f'round{i}_result/challenger_buffer.json')
        final = read(ROOT / 'runs' / f'round{i}_result/summary.json')
        rounds.append({'round': i, **totals([g for g in all_groups if g['run'].startswith(f'round{i}_')]),
                       'optimizer_updates': sum(b['optimizer_updates'] for b in branches if b['run'].startswith(f'round{i}_')),
                       'plan_path': f'runs/{preparation}/plan.json', 'plan_sha256': plan['plan_sha256'],
                       'base_checkpoint': plan['base_checkpoint'], 'training_seed': plan['training_seed'],
                       'archive_cells': [cell['key'] for cell in archive['cells']],
                       'archive_lead_checkpoint': archive['lead']['evaluation']['checkpoint'],
                       'buffer': {k: len(v) if isinstance(v, list) else v for k, v in buffer.items()},
                       'j_plus': final['j_plus'], 'rft_status': final['rft_status']})
    result = {'status': 'journal_and_saved_cost_crosschecks_passed',
              'source_cost_report': 'evidence/COST_THREE_ROUNDS.json', 'source_cost_report_sha256': digest(ROOT / 'evidence/COST_THREE_ROUNDS.json'),
              'definitions': {'full_group': 'One non-drain same-prompt batch of exactly 8 real Solver samples.',
                              'drain': 'Explicit phase-boundary budget remainder generation; not an optimizer group.',
                              'nonzero_advantage_native_tokens': 'Native tokens in mixed binary-reward groups. This is not proof every grammar-forced position has a nonzero gradient.',
                              'loss_eligible_tokens': 'Ledger non-drain token count, including all-constant groups that actually skip the optimizer; do not call this effective learning.',
                              'no_new_generation': True},
              'totals': {**totals(all_groups), 'optimizer_updates': sum(b['optimizer_updates'] for b in branches)},
              'by_task_kind': {kind: totals([g for g in all_groups if g['task_kind'] == kind]) for kind in sorted({g['task_kind'] for g in all_groups})},
              'by_prompt': {p: totals([g for g in all_groups if g['prompt_id'] == p]) for p in sorted({g['prompt_id'] for g in all_groups})},
              'rounds': rounds, 'branches': branches, 'positive_rollouts': positives, 'groups': all_groups,
              'all_rounds_share_one_initial_checkpoint': len({r['base_checkpoint'] for r in rounds}) == 1,
              'all_branches_share_initial_parameter_fingerprint': len({b['initial_parameter_fingerprint'] for b in branches}) == 1,
              'persistent_outer_loop_evidence': {'archive_constructor': 'round/round_cli.py:398',
                                                 'buffer_constructor': 'round/round_cli.py:427',
                                                 'archive_and_buffer_reinitialized_per_scoring_call': True,
                                                 'buffer_zero_positive_rounds_is_one_in_each_independent_result': all(r['buffer']['zero_positive_rounds'] == 1 for r in rounds)}}
    destination = ROOT / 'evidence/result_analysis_training.json'
    assert not destination.exists(), 'Preserve earlier analysis; choose another output name.'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'totals': result['totals'], 'rounds': [{k:v for k,v in r.items() if k not in ('buffer',)} for r in rounds],
                      'by_task_kind': result['by_task_kind'], 'positive_rollouts': positives}, indent=2))


if __name__ == '__main__':
    main()
