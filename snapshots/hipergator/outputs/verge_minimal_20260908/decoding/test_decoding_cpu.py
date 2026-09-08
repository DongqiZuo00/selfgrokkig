"""CPU grammar and constrained-policy tests; synthetic programs are not training data."""
import importlib.util
import math
import unittest

from dsl_grammar import PAINTERS, PULLERS, PROGRAM_PREFIX, build_grammar, example_program


class GrammarSpecificationTests(unittest.TestCase):
    def test_complete_node_count_range_and_only_standard_intermediate_types(self):
        grammar = build_grammar()
        self.assertIn("program0 ::=", grammar)
        self.assertIn("program30 ::=", grammar)
        self.assertNotIn("program31 ::=", grammar)
        for kind in PAINTERS + tuple(PULLERS):
            self.assertIn(kind, grammar)
        self.assertNotIn("RBYGRB", grammar)

    def test_invalid_node_limits_rejected(self):
        for maximum in (1, 33, True):
            with self.assertRaises(ValueError):
                build_grammar(maximum)


HAS_XGRAMMAR = importlib.util.find_spec("xgrammar") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_XGRAMMAR, "xgrammar is in the installed vLLM CPU environment")
class RealXGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        cls.xgr = xgr
        info = xgr.TokenizerInfo([chr(i) for i in range(128)] + ["<eos>"], xgr.VocabType.RAW,
                                vocab_size=129, stop_token_ids=[128])
        cls.compiled = xgr.GrammarCompiler(info, max_threads=2).compile_grammar(build_grammar())

    def accepted(self, source):
        matcher = self.xgr.GrammarMatcher(self.compiled)
        return matcher.accept_string(source) and matcher.accept_token(128) and matcher.is_terminated()

    def test_two_and_thirty_two_nodes_and_every_type_accept(self):
        self.assertTrue(self.accepted(example_program()))
        self.assertTrue(self.accepted(example_program((PAINTERS + tuple(PULLERS)) * 5)))

    def test_static_self_loop_is_allowed(self):
        self.assertTrue(self.accepted(example_program(("PULLER_RB",), self_loop=True)))

    def test_undeclared_routes_duplicate_nodes_and_bad_types_reject(self):
        valid = example_program(("PAINTER_RED", "PULLER_YG"))
        for invalid in (valid.replace("NEXT n0", "NEXT n29", 1),
                        valid.replace("PULLER_YG n1:", "PULLER_YG n0:"),
                        valid.replace("PAINTER_RED", "PAINTER_PURPLE"),
                        valid.replace("[Y]", "[R]")):
            with self.subTest(invalid=invalid):
                self.assertFalse(self.accepted(invalid))

    def test_start_end_and_none_routes_are_available(self):
        for route in ("start", "end", "NONE"):
            self.assertTrue(self.accepted(PROGRAM_PREFIX + route + "\nEND end\n```"))

    def test_token_masks_replay_same_actions_and_reject_early_eos(self):
        matcher = self.xgr.GrammarMatcher(self.compiled)
        self.assertTrue(matcher.accept_string(PROGRAM_PREFIX))
        self.assertFalse(matcher.accept_token(128))
        suffix = example_program(("PAINTER_RED",))[len(PROGRAM_PREFIX):]
        ids = list(map(ord, suffix)) + [128]
        masks = self.xgr.allocate_token_bitmask(len(ids), 129)
        for position, token in enumerate(ids):
            matcher.fill_next_token_bitmask(masks, position)
            self.assertEqual((int(masks[position, token // 32]) >> (token % 32)) & 1, 1)
            self.assertTrue(matcher.accept_token(token))
        self.assertTrue(matcher.is_terminated())


@unittest.skipUnless(HAS_TORCH, "Torch is available in the actual training environment")
class MaskedPolicyTests(unittest.TestCase):
    def test_masked_logp_differs_from_unmasked_and_backprop_respects_mask(self):
        import numpy as np
        import torch
        from hf_grammar_bridge import masked_chosen_logp
        logits = torch.tensor([[1., 2., 99., -4.], [2., 8., 4., 10.]], requires_grad=True)
        targets = torch.tensor([1, 2])
        packed = np.array([[3], [4]], dtype=np.int32)  # {0,1}, then only {2}
        logp = masked_chosen_logp(logits, targets, packed)
        self.assertAlmostEqual(float(logp[0].detach()), 2 - math.log(math.exp(1) + math.exp(2)), places=6)
        self.assertEqual(float(logp[1].detach()), 0)
        (-logp.sum()).backward()
        self.assertEqual(float(logits.grad[0, 2]), 0)
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(4)))
        self.assertNotAlmostEqual(float(logp[0].detach()), float(logits[0].log_softmax(-1)[1].detach()))

    def test_disallowed_actions_and_uncovered_padded_vocab_reject(self):
        import numpy as np
        import torch
        from hf_grammar_bridge import allowed_tensor, masked_chosen_logp
        with self.assertRaises(ValueError):
            masked_chosen_logp(torch.zeros((1, 4)), torch.tensor([2]), np.array([[3]], dtype=np.int32))
        with self.assertRaises(ValueError):
            allowed_tensor(np.array([[3]], dtype=np.int32), 33, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
