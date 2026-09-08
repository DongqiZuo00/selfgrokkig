"""CPU regression checks. Synthetic gradients are tests, never research rewards."""
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from verge_repair_training import phase_boundary_masks, chosen_masked_logp, train_step
from verge_repair_config import repair_config
from verge_repair_protocol import repair_proposal_messages


class IntegrationTests(unittest.TestCase):
    def test_full_groups_and_only_terminal_mask(self):
        self.assertEqual(phase_boundary_masks([2048] * 8, 16384, 42), [[1]*2048]*8)
        masks = phase_boundary_masks([2048]*8, 1000, 42)
        self.assertEqual(sum(map(sum, masks)), 1000)
        self.assertTrue(all(0 < sum(m) < 2048 for m in masks))
        self.assertEqual(masks, phase_boundary_masks([2048]*8, 1000, 42))
        with self.assertRaises(ValueError):
            phase_boundary_masks([2048]*64, 1000, 42)

    def test_complete_fixed_cap_budget_simulation(self):
        from verge_round_core import stage_tokens
        for n in range(1, 5):
            total, boundary_groups, updates = 0, 0, 0
            for quota in stage_tokens(524288, n):
                stage_used = 0
                while stage_used < quota:
                    unit = 0
                    while unit < min(16384, quota-stage_used):
                        lengths = [2048]*8
                        masks = phase_boundary_masks(lengths, quota-stage_used-unit, 42)
                        unit += sum(map(sum, masks))
                        boundary_groups += int(sum(map(sum, masks)) < sum(lengths))
                    stage_used += unit
                    updates += 1
                total += stage_used
            self.assertEqual(total, 524288)
            self.assertLessEqual(boundary_groups, n+1)
            self.assertLessEqual(updates, 100)

    def test_grammar_normalizer_and_gradient(self):
        logits = torch.tensor([[1., 50., 3., -8.], [0., 0., 0., 0.]], requires_grad=True)
        packed = np.array([[5], [10]], dtype=np.int32)  # allow {0,2}, {1,3}
        targets = torch.tensor([2, 1])
        actual = chosen_masked_logp(logits, targets, packed)
        expected = torch.stack([torch.log_softmax(logits[0, [0,2]], 0)[1],
                                torch.log_softmax(logits[1, [1,3]], 0)[0]])
        self.assertTrue(torch.allclose(actual, expected))
        (-actual.mean()).backward()
        self.assertEqual(logits.grad[0,1].item(), 0.)
        self.assertNotEqual(logits.grad[0,0].item(), 0.)
        with self.assertRaises(RuntimeError):
            chosen_masked_logp(logits, torch.tensor([1,1]), packed)

    def test_bit31_and_full_support(self):
        logits = torch.arange(33, dtype=torch.float).reshape(1,33)
        packed = np.array([[-2147483648, 0]], dtype=np.int32)
        self.assertEqual(chosen_masked_logp(logits, torch.tensor([31]), packed).item(), 0.)
        got = chosen_masked_logp(logits, torch.tensor([20]))
        self.assertTrue(torch.allclose(got, torch.log_softmax(logits, -1)[:,20]))

    def test_real_gradient_accumulation_and_zero_stall(self):
        from types import SimpleNamespace
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4,4))
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weight[input_ids])
        model = Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        records = [{"prompt_token_ids": [0], "completion_token_ids": [1,2]} for _ in range(8)]
        masks = [[1,1]]*8
        initial = model.weight.detach().clone()
        zero = train_step(model, None, optimizer, scheduler, records, [0.]*8, masks, {"kl_beta":0.}, 0)
        self.assertFalse(zero["optimizer_step"])
        self.assertTrue(torch.equal(model.weight, initial))
        # A fixed synthetic action group tests code, not curriculum value.
        records[0]["completion_token_ids"] = [3,3]
        step = train_step(model, None, optimizer, scheduler, records, [1.]+[-1/7]*7, masks, {"kl_beta":0.}, 0)
        self.assertTrue(step["optimizer_step"])
        self.assertEqual(step["tokens_used"], 16)
        self.assertFalse(torch.equal(model.weight, initial))

    def test_isolated_config_and_compact_history(self):
        exp = Path(__file__).resolve().parents[1]
        cfg = repair_config("verge_mistral_repair_v2", exp)
        self.assertEqual(cfg["scope_max_drop"], 0.)
        self.assertEqual(cfg["completion_tokens"], 2048)
        self.assertEqual(cfg["train_tokens_per_branch"], 524288)
        self.assertFalse(cfg["acceptance_only"])
        target = {"counts":[0]*11, "rates":[0]*11, "rollouts":512,
                  "condition_names":["c"]*11, "bottleneck_index":0,
                  "keyed_vectors":{"large":"must not enter prompt"}}
        messages = repair_proposal_messages("target", {"interface_version":"verge_code_prefix_v2", "target":target}, {}, {})
        self.assertNotIn("keyed_vectors", json.dumps(messages))
        self.assertNotIn("must not enter prompt", json.dumps(messages))

    def test_native_actions_and_prompt_identity(self):
        from verge_repair_sampling import require_native_choice
        value = {"token_ids": [4,5,2], "prompt_token_ids": [1,3]}
        self.assertEqual(require_native_choice(value, [1,3]), [4,5,2])
        with self.assertRaises(RuntimeError):
            require_native_choice({"text":"never retokenize me"}, [1,3])
        with self.assertRaises(RuntimeError):
            require_native_choice(value, [1,9])

    def test_zero_update_identity_not_monte_carlo_credit(self):
        from test_verge_runtime import RuntimeTest, endpoint
        from verge_round_decision import decide
        frozen = RuntimeTest().frozen()
        frozen["config"].update(repair_integrated=True, scope_rule="observed zero drop")
        branches = {i:endpoint(0, changed=False) for i in range(4)}
        branches[1] = endpoint(1, changed=False)  # Simulated sampling noise, same unchanged weights.
        result = decide(frozen, branches)
        self.assertEqual(result["challenger_rewards"], [0.,0.,0.])
        self.assertEqual(result["challenger_advantages"], [0.,0.,0.])
        self.assertFalse(result["challenger_update_allowed"])
        self.assertFalse(result["evaluated_candidates"][0]["eligible"])


if __name__ == "__main__":
    unittest.main()
