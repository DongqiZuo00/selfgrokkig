"""Bounded continuation configuration and launch gates; no Slurm/model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from verge_followon_protocol import (candidate_config,candidate_from_bank,validate_plan,validate_launch)
from verge_followon_sources import cohort_sources
from test_verge_followon_sources import fixture

EXP = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((EXP/'manifests/verge_followon_v1_protocol.json').read_text())


def bank():
    return {'protocol':'verge_followon_v1_source_bank','candidate_count':72,
        'cohorts':[cohort_sources(*fixture(arm,r))
                   for r in range(6) for arm in ('verge','frozen','uncertainty','outcome')]}


class ProtocolTests(unittest.TestCase):
    def test_all_72_unique_bounded_configurations(self):
        names = set(); b,p = bank(),plan()
        for i in range(72):
            cfg = candidate_config(EXP,b,i,p)
            names.add(cfg['protocol_version'])
            self.assertEqual(cfg['train_tokens_per_branch'],262144)
            self.assertEqual(cfg['followon_phases'],[{'kind':'target','reward_mode':'binary','tokens':262144}])
            self.assertFalse(cfg.get('book_suite'))
            self.assertFalse(cfg.get('control_only'))
            self.assertFalse(cfg['stop_after_first_success'])
            self.assertEqual(cfg['solver_start'],cfg['followon_source']['source_checkpoint'])
        self.assertEqual(len(names),72)

    def test_paired_within_cohort_only(self):
        configs=[candidate_config(EXP,bank(),i,plan()) for i in range(4)]
        self.assertEqual(len({c['random_stream_id'] for c in configs[:3]}),1)
        self.assertNotEqual(configs[2]['random_stream_id'],configs[3]['random_stream_id'])

    def test_budget_reward_seed_and_resource_changes_rejected(self):
        for key,value in (('loss_tokens_per_candidate',524288),('training_seed',43),
                          ('solver_reward','dense'),('maximum_total_gpus',3),
                          ('optimizer_reset','inherit'),('early_stop',True)):
            p=plan();p[key]=value
            with self.assertRaises(ValueError):validate_plan(p)

    def test_missing_censoring_plan_rejected(self):
        p=plan();del p['censoring']
        with self.assertRaises(ValueError):validate_plan(p)

    def test_partial_reordered_or_foreign_bank_rejected(self):
        for kind in ('partial','reordered','foreign'):
            b=bank()
            if kind == 'partial':b['cohorts'].pop()
            elif kind == 'reordered':b['cohorts'][0],b['cohorts'][1]=b['cohorts'][1],b['cohorts'][0]
            else:b['cohorts'][0]['candidates'][0]['source_cohort']='foreign'
            with self.assertRaises(ValueError):candidate_from_bank(b,0)

    def test_source_checkpoint_cannot_be_substituted(self):
        b=bank();b['cohorts'][0]['candidates'][0]['source_checkpoint']='checkpoints/other/resume_u0030'
        with self.assertRaises(ValueError):candidate_config(EXP,b,0,plan())

    def test_factory_does_not_mutate_source(self):
        b,p=bank(),plan();original=copy.deepcopy((b,p))
        cfg=candidate_config(EXP,b,0,p);cfg['followon_source']['curriculum_spec']['stages'].clear()
        self.assertEqual((b,p),original)

    def test_no_launch_before_freeze_and_predecessors(self):
        b,p=bank(),plan();cfg=candidate_config(EXP,b,0,p)
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);root=exp/'raw_results/verge_followon_v1'
            with self.assertRaisesRegex(RuntimeError,'frozen plan'):validate_launch(exp,cfg)
            write(exp/'manifests/verge_followon_v1_protocol.json',p)
            write(root/'protocol_frozen.json',p);write(root/'source_bank.json',b);write(root/'source_bank_frozen.json',b)
            with self.assertRaisesRegex(RuntimeError,'predecessor'):validate_launch(exp,cfg)
            write(exp/'raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json',{'primary_block_complete':True,'outer_rounds':24})
            write(exp/'raw_results/verge_control_v1/DIRECT_CONTROL_BLOCK_COMPLETE.json',{'control_block_complete':True})
            write(exp/'manifests/verge_mistral_round1.json',json.loads((EXP/'manifests/verge_mistral_round1.json').read_text()))
            checkpoint=exp/cfg['solver_start'];write(checkpoint/'verge_committed.json',{'update':30})
            for name in ('state.pt','adapter_model.safetensors'):(checkpoint/name).write_bytes(b'synthetic-not-model-weights')
            self.assertEqual(validate_launch(exp,cfg),p)
            changed=copy.deepcopy(cfg);changed['kl_beta']=0.
            with self.assertRaisesRegex(RuntimeError,'intervention'):validate_launch(exp,changed)
            changed=copy.deepcopy(p);changed['tie_rule']='different'
            write(root/'protocol_frozen.json',changed)
            with self.assertRaisesRegex(RuntimeError,'changed after'):validate_launch(exp,cfg)


if __name__ == '__main__':unittest.main()
