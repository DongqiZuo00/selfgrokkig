import json
import tempfile
import unittest
from pathlib import Path

from generated_budget import (JsonlJournal, PhaseLedger, Rollout, assert_fresh_optimizer,
                              binary_advantages, run_phase, verify_journal)


class GeneratedBudgetTests(unittest.TestCase):
    def execute(self, quota, pattern, reward_pattern):
        with tempfile.TemporaryDirectory() as tmp:
            journal = JsonlJournal(Path(tmp) / "events.jsonl", "synthetic-budget-test")
            updates = []
            def generate(n, cap, drain):
                return [Rollout("same-prompt", "target", tuple([3] * min(cap, pattern[i % len(pattern)])),
                                reward_pattern[i % len(reward_pattern)], "fixture", "synthetic-verifier")
                        for i in range(n)]
            ledger = run_phase(PhaseLedger("stage1", quota), generate=generate,
                               update=lambda rows, adv: updates.append((rows, adv)), journal=journal)
            self.assertGreater(verify_journal(journal.path), 0)
            return ledger, updates

    def test_actual_generation_exact_with_early_eos_and_boundary_drain(self):
        ledger, updates = self.execute(101, [1, 2, 3, 5, 8], [0, 1])
        self.assertEqual(ledger.generated_tokens, 101)
        self.assertGreater(ledger.boundary_drain_tokens, 0)
        self.assertLessEqual(ledger.boundary_drain_tokens, 7)
        self.assertEqual(ledger.generated_tokens,
                         ledger.loss_eligible_tokens + ledger.boundary_drain_tokens)
        self.assertEqual(ledger.optimizer_steps, len(updates))
        self.assertEqual(ledger.nonzero_advantage_tokens, ledger.loss_eligible_tokens)

    def test_equal_direct_and_curriculum_generated_budget(self):
        direct, _ = self.execute(131, [1], [0])
        curriculum, _ = self.execute(131, [1, 1, 1, 1, 1, 1, 1, 2], [0, 1])
        self.assertEqual(direct.generated_tokens, curriculum.generated_tokens)
        self.assertNotEqual(direct.loss_eligible_tokens, curriculum.loss_eligible_tokens)
        self.assertEqual(direct.optimizer_steps, 0)

    def test_all_zero_and_all_one_do_not_call_update(self):
        for reward in (0, 1):
            ledger, updates = self.execute(83, [2], [reward])
            self.assertEqual(updates, [])
            self.assertEqual(ledger.nonzero_advantage_tokens, 0)
            self.assertGreater(ledger.constant_groups, 0)

    def test_binary_group_contract(self):
        for rewards in ([0] * 7, [0.2] * 8, [True] * 8):
            with self.assertRaises(ValueError):
                binary_advantages(rewards)
        self.assertEqual(binary_advantages([0] * 8), (0.,) * 8)
        self.assertAlmostEqual(sum(binary_advantages([0, 1] * 4)), 0)

    def test_backends_cannot_overrun_cap_or_mix_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, bad in (("overrun", lambda i: Rollout("x", "target", (1, 2), 0, "", "v")),
                               ("mixed", lambda i: Rollout(str(i), "target", (1,), 0, "", "v"))):
                with self.assertRaises(ValueError):
                    run_phase(PhaseLedger(label, 8),
                              generate=lambda n, cap, drain: [bad(i) for i in range(n)],
                              update=lambda rows, adv: self.fail("unexpected update"),
                              journal=JsonlJournal(Path(tmp) / (label + ".jsonl"), label))

    def test_log_tampering_and_existing_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            journal = JsonlJournal(path, "synthetic")
            journal.append("x", {"tokens": 4})
            self.assertEqual(verify_journal(path), 1)
            with self.assertRaises(FileExistsError):
                JsonlJournal(path, "overwrite")
            path.write_text(path.read_text().replace('"tokens": 4', '"tokens": 5'))
            with self.assertRaises(ValueError):
                verify_journal(path)

    def test_fresh_actual_optimizer_settings(self):
        class Optimizer:
            state = {}
            param_groups = [{"weight_decay": 0}]
        optimizer = Optimizer()
        assert_fresh_optimizer(optimizer, beta=0, weight_decay=0)
        optimizer.state = {"momentum": 1}
        with self.assertRaises(ValueError):
            assert_fresh_optimizer(optimizer, beta=0, weight_decay=0)

    def test_partial_ledger_requires_explicit_resume_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_phase(PhaseLedger("bad-resume", 8, generated_tokens=-1),
                          generate=lambda *args: self.fail("must fail before generation"),
                          update=lambda *args: None,
                          journal=JsonlJournal(Path(tmp) / "log.jsonl", "synthetic"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
