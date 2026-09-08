"""CPU-only real Torch update tests; TinyLM is explicitly a synthetic fixture.

Run where Torch is available: python -B -m unittest discover -s runtime
-p test_torch_solver.py -v. No model download, GPU, or old checkpoint is used.
"""
import copy
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from generated_budget import binary_advantages
from solver_update import binary_solver_train_step, make_fresh_solver_branch, train_step


if torch is not None:
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 8)
            self.dropout = torch.nn.Dropout(0.25)
            self.lm_head = torch.nn.Linear(8, 13)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(logits=self.lm_head(self.dropout(self.embedding(input_ids))))


def sample_items():
    # Success and failure groups choose different tokens, so gradients need not
    # cancel under population-normalized +1/-1 advantages.
    return [{"prompt_token_ids": [1, 2], "completion_token_ids": [3, 4] if i < 4 else [5, 6]}
            for i in range(8)]


@unittest.skipIf(torch is None, "Torch is unavailable locally; this test must run on remote CPU")
class TorchSolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.base = TinyLM().cpu()
        self.model, self.optimizer, self.scheduler = make_fresh_solver_branch(self.base, learning_rate=0.01)
        self.items = sample_items()

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def warm_optimizer(self):
        result = binary_solver_train_step(self.model, self.optimizer, self.scheduler,
                                          self.items, [1] * 4 + [0] * 4)
        self.assertTrue(result["optimizer_step"])
        self.assertTrue(self.optimizer.state)
        return result

    def test_binary_advantages_constants_exact_zero_mixed_normalized(self):
        self.assertEqual(binary_advantages([0] * 8), (0.,) * 8)
        self.assertEqual(binary_advantages([1] * 8), (0.,) * 8)
        mixed = binary_advantages([1] * 4 + [0] * 4)
        self.assertAlmostEqual(sum(mixed), 0.)
        self.assertGreater(mixed[0], 0.)
        self.assertLess(mixed[-1], 0.)
        with self.assertRaises(ValueError):
            binary_advantages([0.5] * 8)

    def test_real_review_train_step_skips_warm_adamw_all_zero_and_one(self):
        self.warm_optimizer()
        before_weights = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        before_scheduler = copy.deepcopy(self.scheduler.state_dict())
        for reward in (0, 1):
            result = train_step(self.model, None, self.optimizer, self.scheduler,
                                self.items, binary_advantages([reward] * 8), [[1, 1] for _ in range(8)],
                                {"kl_beta": 0.}, prior_steps=7)
            self.assertFalse(result["optimizer_step"])
            self.assertEqual(result["nonzero_advantage_tokens"], 0)
            self.assertEqual(result["tokens_used"], 16)
            self.assertEqual(result["loss"], 0.0)
            self.assert_tree_equal(before_weights, self.model.state_dict())
            self.assert_tree_equal(before_optimizer, self.optimizer.state_dict())
            self.assert_tree_equal(before_scheduler, self.scheduler.state_dict())

    def test_mixed_group_moves_parameters_and_scheduler(self):
        before_weights = copy.deepcopy(self.model.state_dict())
        before_epoch = self.scheduler.last_epoch
        result = self.warm_optimizer()
        self.assertTrue(any(not torch.equal(before_weights[name], value) for name, value in self.model.state_dict().items()))
        self.assertEqual(self.scheduler.last_epoch, before_epoch + 1)
        self.assertEqual(result["tokens_used"], 16)
        self.assertEqual(result["nonzero_advantage_tokens"], 16)
        self.assertEqual(self.model.dropout.p, 0.)

    def test_each_branch_clones_current_base_weights_but_fresh_optimizer(self):
        self.warm_optimizer()
        first, opt1, sched1 = make_fresh_solver_branch(self.model, learning_rate=0.01)
        second, opt2, sched2 = make_fresh_solver_branch(self.model, learning_rate=0.01)
        self.assert_tree_equal(first.state_dict(), self.model.state_dict())
        self.assert_tree_equal(second.state_dict(), self.model.state_dict())
        self.assertFalse(opt1.state)
        self.assertFalse(opt2.state)
        self.assertIsNot(opt1, opt2)
        self.assertEqual(sched1.last_epoch, sched2.last_epoch)
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertNotEqual(p.data_ptr(), q.data_ptr())
        binary_solver_train_step(first, opt1, sched1, self.items, [1] * 4 + [0] * 4)
        self.assertTrue(opt1.state)
        self.assertFalse(opt2.state)
        self.assert_tree_equal(second.state_dict(), self.model.state_dict())

    def test_operational_entry_forbids_beta_weight_decay_and_partial_groups(self):
        self.assertEqual(self.optimizer.param_groups[0]["weight_decay"], 0.)
        with self.assertRaises(ValueError):
            binary_solver_train_step(self.model, self.optimizer, self.scheduler, self.items, [0] * 8, beta=0.1)
        self.optimizer.param_groups[0]["weight_decay"] = 0.01
        with self.assertRaises(ValueError):
            binary_solver_train_step(self.model, self.optimizer, self.scheduler, self.items, [0] * 8)
        self.optimizer.param_groups[0]["weight_decay"] = 0.
        with self.assertRaises(ValueError):
            binary_solver_train_step(self.model, self.optimizer, self.scheduler, self.items[:4], [0] * 4)
        self.assertFalse(self.optimizer.state)

    def test_direct_branch_before_success_exactly_equals_base(self):
        before_scheduler = copy.deepcopy(self.scheduler.state_dict())
        for _ in range(5):
            result = binary_solver_train_step(self.model, self.optimizer, self.scheduler,
                                              self.items, [0] * 8)
            self.assertFalse(result["optimizer_step"])
        self.assert_tree_equal(self.base.state_dict(), self.model.state_dict())
        self.assertFalse(self.optimizer.state)
        self.assert_tree_equal(before_scheduler, self.scheduler.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
