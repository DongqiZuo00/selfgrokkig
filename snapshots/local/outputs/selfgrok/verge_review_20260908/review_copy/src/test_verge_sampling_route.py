"""Fake HTTP integration checks for the production sampler, with no model draws."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_repair_sampling as sampler
from verge_independent_sampling import PROTOCOL


class SamplerRouteTests(unittest.TestCase):
    def run_batch(self, path, independent, calls):
        client = SimpleNamespace(model_name="fixed", base_url="http://fake", gpu=0,
                                 verifier_workers=2, tokenizer=None)
        rows = [{"id": str(i), "messages": [], "ground_truth": []} for i in range(3)]
        def http(url, req, timeout):
            calls.append(req)
            return {"model": req["model"], "choices": [
                {"index": i, "token_ids": [5,6], "prompt_token_ids": [1,2], "text": "END end"}
                for i in range(len(req["prompt"])*req["n"])],
                "usage": {"prompt_tokens": 2*len(req["prompt"]),
                          "completion_tokens": 2*len(req["prompt"])*req["n"],
                          "total_tokens": 2*len(req["prompt"])*(1+req["n"])}}
        with patch.object(sampler, "solver_prompt", return_value=("prompt", [1,2])), \
             patch.object(sampler, "http_json", side_effect=http), \
             patch.object(sampler, "TelemetrySampler", side_effect=lambda *a: nullcontext()), \
             patch.object(sampler, "ProcessPoolExecutor", ThreadPoolExecutor), \
             patch.object(sampler, "_verify_payload", return_value={"reward": 0}):
            return sampler.score_rows(client, rows, 8, 42, path,
                                      independent_prompt_streams=independent)

    def test_new_route_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path, calls = Path(directory)/"target.jsonl", []
            records = self.run_batch(path, True, calls)
            self.assertEqual(len(calls), 3)
            self.assertEqual(len({r["sampling_child_seed"] for r in records}), 24)
            self.assertTrue(all(r["sampling_protocol"] == PROTOCOL for r in records))
            self.assertEqual(self.run_batch(path, True, calls), records)
            self.assertEqual(len(calls), 3)
            stored = json.loads(path.with_suffix(".response.json").read_text())
            self.assertEqual(stored["response"]["child_seed_count"], 24)

    def test_legacy_route_unchanged_and_cannot_be_reused_by_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path, calls = Path(directory)/"target.jsonl", []
            records = self.run_batch(path, False, calls)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]["prompt"]), 3)
            self.assertNotIn("sampling_protocol", records[0])
            with self.assertRaises(RuntimeError):
                self.run_batch(path, True, calls)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
