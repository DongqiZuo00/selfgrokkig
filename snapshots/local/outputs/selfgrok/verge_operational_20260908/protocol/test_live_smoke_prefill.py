"""CPU contract checks for the failure-driven GPU entry; no model is loaded."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
import live_smoke_prefill as smoke


class PrefillInterfaceTests(unittest.TestCase):
    def test_both_prompt_routes_append_format_prefix_once(self):
        class PlainTokenizer:
            chat_template = None
            bos_token = "<s>"
        class ChatTokenizer(PlainTokenizer):
            chat_template = "fixture"
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "frozen-chat-user-prompt"
        messages = [{"role": "user", "content": "SYNTHETIC target description"}]
        for tokenizer in (PlainTokenizer(), ChatTokenizer()):
            text = smoke.prompt_text(tokenizer, messages)
            self.assertTrue(text.endswith(smoke.PROGRAM_PREFIX))
            self.assertEqual(text.count(smoke.PROGRAM_PREFIX), 1)

    def test_verifier_receives_prefix_plus_native_suffix(self):
        suffix = "end\nEND end\n```"
        program = smoke.full_program(suffix)
        self.assertEqual(program, smoke.PROGRAM_PREFIX + suffix)
        self.assertTrue(program.startswith("```manufactoria\nSTART start:\n    NEXT end"))
        self.assertEqual(suffix, "end\nEND end\n```")

    def test_production_prefix_is_checked_without_executing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.py"
            source.write_text("raise RuntimeError('must never execute')\nPROGRAM_PREFIX = " + repr(smoke.PROGRAM_PREFIX), encoding="utf-8")
            smoke.verify_format_source(source)
            source.write_text("PROGRAM_PREFIX = 'wrong'", encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke.verify_format_source(source)

    def test_generated_tokens_keep_eos_and_only_remove_post_eos_padding(self):
        self.assertEqual(smoke.trim_completion([7, 2, 0, 0], {2}), [7, 2])
        self.assertEqual(smoke.trim_completion([7, 2, 2, 2], {2}), [7, 2])
        self.assertEqual(smoke.trim_completion([7, 0, 7], {2}), [7, 0, 7])

    def test_budget_sampling_loss_interface_and_sbatch_match(self):
        self.assertEqual((smoke.GROUP_COUNT, smoke.GROUP_SIZE, smoke.MAX_NEW_TOKENS), (1, 8, 2048))
        self.assertEqual(smoke.MAX_GENERATED_TOKENS, 16384)
        self.assertEqual(smoke.GPU_MINUTES_CEILING, 17)
        text = (ROOT / "runtime/live_smoke_prefill.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        generation = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"]
        self.assertEqual(len(generation), 1)
        keywords = {kw.arg: kw.value for kw in generation[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["temperature"]), 1.0)
        self.assertEqual(ast.literal_eval(keywords["top_p"]), 1.0)
        self.assertEqual(ast.literal_eval(keywords["top_k"]), 0)
        self.assertNotIn("stop_strings", keywords)
        # Training items are formed from native generated token lists, not from
        # re-tokenized full_program strings. The latter are only verifier input.
        self.assertIn('"completion_token_ids": ids} for ids in completions]', text)
        self.assertIn('for program in verifier_completions]', text)
        sbatch = (ROOT / "RUN_GPU_SMOKE_PREFILL.sbatch").read_bytes()
        self.assertTrue(sbatch.startswith(b"#!/bin/bash\n"))
        self.assertNotIn(b"\r", sbatch)
        self.assertIn(b"#SBATCH --time=00:17:00", sbatch)
        self.assertIn(b"#SBATCH --gpus=b200:1", sbatch)
        self.assertIn(b"#SBATCH --mem=64gb", sbatch)
        self.assertIn(b"live_smoke_prefill.py", sbatch)
        self.assertIn(b"gpu_smoke_prefill_${SLURM_JOB_ID}", sbatch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
