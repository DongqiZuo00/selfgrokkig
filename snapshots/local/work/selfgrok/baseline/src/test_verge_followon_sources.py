"""Synthetic source-bank tests: no model, sampling, checkpoint loads or hashes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verge_followon_sources as sources


def fixture(arm='verge', round_index=0):
    name = f'verge_book_v2_{arm}_r{round_index:02d}'
    cfg = {'book_suite':'verge_book_v2','independent_prompt_streams':True,
        'sampling_protocol':'per_prompt_disjoint_seed_ranges_v1','seed':42,
        'completion_tokens':2048,'train_tokens_per_branch':524288,
        'book_arm':arm,'book_round':round_index,'protocol_version':name,
        'solver_start':'checkpoints/verge_book_v2_initial_solver/resume_u0000'}
    frozen = {'config':cfg,'generation':{'candidates':[
        {'index':i,'valid':True,'spec':{'stages':[{'family':'synthetic_only'}]}}
        for i in (1,2,3)]}}
    branches = {}
    for i in range(4):
        counts = [10+i*5]+[0]*10
        branches[i] = {'checkpoint':f'checkpoints/{name}_{"direct" if i == 0 else "candidate_"+str(i)}/resume_u0030',
            'target':{'rollouts':512,'counts':counts,'rates':[n/512 for n in counts]},
            'training':{'train_tokens':524288,'update':30,'optimizer_steps':i,
                'target_successes':0,'first_target_success':None,
                'curriculum_stage_reward_counts':[] if i == 0 else [{'draws':100,'successes':10*i}]}}
    decision = {'target_condition_gains':[5/512,10/512,15/512],
        'reward_rung_zero_based':0,'endpoint_only_credit':True,
        'selected':{'name':'candidate_1'},'evaluated_candidates':[
            {'index':i,'scope_safe':i != 2,'eligible':i == 1} for i in (1,2,3)]}
    frozen['initial'] = copy.deepcopy(branches[0])
    return frozen,decision,branches


class SourceTests(unittest.TestCase):
    def test_retains_unsafe_ineligible_and_unselected(self):
        source = sources.cohort_sources(*fixture())
        self.assertEqual([c['candidate_index'] for c in source['candidates']],[1,2,3])
        self.assertFalse(source['candidates'][1]['scope_safe_observed'])
        self.assertFalse(source['candidates'][2]['was_eligible'])
        self.assertEqual(sum(c['was_selected'] for c in source['candidates']),1)
        self.assertTrue(all(c['follow_on_loss_tokens'] == 262144 for c in source['candidates']))

    def test_zero_update_credit_keeps_observed_difference_and_candidate(self):
        frozen,decision,branches = fixture()
        branches[2]['training']['optimizer_steps'] = 0
        decision['target_condition_gains'][1] = 0
        c = sources.cohort_sources(frozen,decision,branches)['candidates'][1]
        self.assertTrue(c['zero_update_policy_identity'])
        self.assertEqual(c['condition_gain'],0)
        self.assertEqual(c['observed_endpoint_condition_gain'],10/512)

    def test_credit_is_not_uncertainty_teacher_reward(self):
        frozen,decision,branches = fixture('uncertainty',2)
        decision['challenger_rewards'] = [999,999,999]
        c = sources.cohort_sources(frozen,decision,branches)['candidates'][0]
        self.assertEqual(c['condition_gain'],5/512)
        self.assertAlmostEqual(c['uncertainty_score'],.2)

    def test_missing_stage_observations_are_not_zero(self):
        for stats in ([],[{'draws':0,'successes':0}],[{'draws':1,'successes':2}],
                      [{'draws':1.5,'successes':1}]):
            with self.subTest(stats=stats):
                frozen,decision,branches = fixture()
                branches[1]['training']['curriculum_stage_reward_counts'] = stats
                with self.assertRaises(RuntimeError):
                    sources.cohort_sources(frozen,decision,branches)

    def test_contract_changes_rejected(self):
        for key,value in (('book_suite','verge_book_v1'),('seed',43),('completion_tokens',4096),
                          ('train_tokens_per_branch',1048576),('independent_prompt_streams',False),
                          ('sampling_protocol','shared_old_stream')):
            with self.subTest(key=key):
                frozen,decision,branches = fixture()
                frozen['config'][key] = value
                with self.assertRaises(RuntimeError):
                    sources.cohort_sources(frozen,decision,branches)

    def test_non_endpoint_or_altered_credit_rejected(self):
        for key,value in (('endpoint_only_credit',False),('reward_rung_zero_based',11),
                          ('target_condition_gains',[1,1,1])):
            frozen,decision,branches = fixture()
            decision[key] = value
            with self.assertRaises(RuntimeError):
                sources.cohort_sources(frozen,decision,branches)

    def test_invalid_profile_rejected(self):
        for change in ('rate','nested','rollouts'):
            frozen,decision,branches = fixture()
            target = branches[1]['target']
            if change == 'rate': target['rates'][0] = .5
            elif change == 'nested': target['counts'][1] = 100; target['rates'][1] = 100/512
            else: target['rollouts'] = 128
            with self.assertRaises(RuntimeError):
                sources.cohort_sources(frozen,decision,branches)

    def test_wrong_checkpoint_or_unfinished_budget_rejected(self):
        for change in ('checkpoint','budget','update','optimizer'):
            frozen,decision,branches = fixture()
            if change == 'checkpoint': branches[2]['checkpoint'] = branches[1]['checkpoint']
            elif change == 'budget': branches[2]['training']['train_tokens'] -= 1
            elif change == 'update': branches[2]['training']['update'] = 29
            else: branches[2]['training']['optimizer_steps'] = -1
            with self.assertRaises(RuntimeError):
                sources.cohort_sources(frozen,decision,branches)

    def test_source_not_mutated(self):
        original = fixture()
        before = copy.deepcopy(original)
        result = sources.cohort_sources(*original)
        result['candidates'][0]['curriculum_spec']['stages'].clear()
        self.assertEqual(original,before)

    def test_no_winner_only_bank_when_primary_incomplete(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sources.book,'EXP',Path(directory)), \
                patch.object(sources.book,'SUITE','verge_book_v2'), \
                patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2'}):
            with self.assertRaisesRegex(RuntimeError,'all 24'):
                sources.collect()
            self.assertFalse((Path(directory)/'raw_results/verge_followon_v1').exists())

    def test_explicit_suite_required(self):
        with patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v1'}):
            with self.assertRaisesRegex(RuntimeError,'Explicit'):
                sources.collect()

    def test_all_72_candidates_and_24_metadata_references(self):
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            write(exp/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json',
                  {'primary_block_complete':True,'outer_rounds':24})
            write(exp/'raw_results/verge_book_v2/protocol_frozen.json',{'follow_on_tokens_per_candidate':262144})
            for arm,r in sources.book.schedule():
                frozen,decision,branches = fixture(arm,r)
                root = exp/'raw_results'/frozen['config']['protocol_version']
                write(root/'round_frozen.json',frozen)
                write(root/'decision.json',decision)
                write(root/'completion_profile.json',{'rollouts':64,'counts':[0]*11})
                for i,branch in branches.items():
                    write(root/'branches'/str(i)/'complete.json',branch)
                    checkpoint = exp/branch['checkpoint']
                    write(checkpoint/'verge_committed.json',{'update':30})
                    # Presence-only placeholders, never interpreted as tensors.
                    for name in ('state.pt','adapter_model.safetensors'):
                        (checkpoint/name).write_bytes(b'synthetic-not-model-weights')
            with patch.object(sources.book,'EXP',exp),patch.object(sources.book,'SUITE','verge_book_v2'), \
                    patch.object(sources.book,'verified_round') as verified, \
                    patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2'}):
                result = sources.collect()
                self.assertEqual(verified.call_count,24)
                self.assertEqual(result['candidate_count'],72)
                self.assertEqual(result['reference_metadata_count'],24)
                self.assertEqual(result['follow_on_candidate_loss_tokens'],18874368)
                self.assertFalse(result['follow_on_training_submitted'])
                self.assertFalse(result['suite_complete'])
                self.assertFalse(result['checkpoint_identity_deduplication'])
                self.assertTrue(all(c['no_full_event_recorded_in_source_cohort'] for c in result['cohorts']))
                first_root = exp/'raw_results/verge_book_v2_verge_r00'
                write(first_root/'completion_profile.json',{'rollouts':64,'counts':[1]*11})
                event_result = sources.collect()
                self.assertFalse(event_result['cohorts'][0]['no_full_event_recorded_in_source_cohort'])
                self.assertEqual(event_result['candidate_count'],72)
                checkpoint = exp/fixture()[2][2]['checkpoint']/'verge_committed.json'
                write(checkpoint,{'update':29})
                with self.assertRaisesRegex(RuntimeError,'commit and update'):
                    sources.collect()


if __name__ == '__main__':
    unittest.main()
