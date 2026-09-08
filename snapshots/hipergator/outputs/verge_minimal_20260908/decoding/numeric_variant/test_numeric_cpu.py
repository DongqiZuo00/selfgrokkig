"""CPU checks of numeric syntax; these fixtures are never Solver samples."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from numeric_grammar import PROGRAM_PREFIX, PAINTERS, PULLERS, build_grammar, rename_identifiers
from dsl_grammar import example_program


class SourceAndLimits(unittest.TestCase):
    def test_frozen_sources_remain_unchanged(self):
        here = Path(__file__).resolve().parent
        expected = json.loads((here / 'FROZEN_SOURCE_GUARDS.json').read_text())['frozen_sha256']
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((here.parent / name).read_bytes()).hexdigest(), digest)

    def test_limits_preserve_two_through_thirty_two_nodes(self):
        for value in (1, 33, True):
            with self.assertRaises(ValueError):
                build_grammar(value)
        self.assertIn('program30 ::=', build_grammar(32))
        self.assertNotIn('program31 ::=', build_grammar(32))


@unittest.skipUnless(importlib.util.find_spec('xgrammar'), 'actual vLLM CPU environment required')
class RealNumericGrammar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        cls.xgr = xgr
        info = xgr.TokenizerInfo([chr(i) for i in range(128)] + ['<eos>'],
                                xgr.VocabType.RAW, vocab_size=129, stop_token_ids=[128])
        cls.compiled = xgr.GrammarCompiler(info, max_threads=2).compile_grammar(build_grammar())

    def accepted(self, source):
        matcher = self.xgr.GrammarMatcher(self.compiled)
        return matcher.accept_string(source) and matcher.accept_token(128) and matcher.is_terminated()

    def test_bounds_every_standard_type_and_self_loops(self):
        for count in (0, 1, 2, 6, 30):
            kinds = PAINTERS + tuple(PULLERS)
            for loop in (False, True):
                source = example_program(tuple(kinds[i % 6] for i in range(count)), self_loop=loop)
                self.assertTrue(self.accepted(rename_identifiers(source)))

    def test_forward_backward_start_end_and_none_references_remain(self):
        source = rename_identifiers(example_program(('PULLER_RB', 'PAINTER_RED', 'PULLER_YG')))
        for route in ('start', 'end', 'NONE', '0', '1', '2'):
            changed = source.replace('NEXT 0', 'NEXT ' + route, 1).replace('[Y] end', '[Y] 0')
            self.assertTrue(self.accepted(changed))

    def test_undeclared_duplicate_wrong_spelling_and_bad_types_reject(self):
        source = rename_identifiers(example_program(('PAINTER_RED', 'PULLER_YG')))
        for invalid in (source.replace('NEXT 0', 'NEXT 29', 1),
                        source.replace('PULLER_YG 1:', 'PULLER_YG 0:'),
                        source.replace('NEXT 0', 'NEXT n0', 1),
                        source.replace('NEXT 0', 'NEXT 00', 1),
                        source.replace('PAINTER_RED', 'PAINTER_PURPLE')):
            self.assertFalse(self.accepted(invalid))

    def test_first_mask_accepts_numeric_names_and_keeps_terminals(self):
        matcher = self.xgr.GrammarMatcher(self.compiled)
        self.assertTrue(matcher.accept_string(PROGRAM_PREFIX))
        mask = self.xgr.allocate_token_bitmask(1, 129)
        matcher.fill_next_token_bitmask(mask)
        allowed = lambda token: bool((int(mask[0, token // 32]) >> (token % 32)) & 1)
        for char in ('0', '1', '2', '9', 's', 'e', 'N'):
            self.assertTrue(allowed(ord(char)))
        for char in ('n', 'p'):
            self.assertFalse(allowed(ord(char)))
        self.assertFalse(allowed(128))


if __name__ == '__main__':
    unittest.main(verbosity=2)
