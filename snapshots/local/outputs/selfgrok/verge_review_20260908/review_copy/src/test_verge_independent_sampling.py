import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from verge_independent_sampling import split_requests, merge_responses, complete, validate_response, validate_records, PROTOCOL


class IndependentSamplingTests(unittest.TestCase):
    def request(self, rows=64, n=8):
        return {"model": "fixed_adapter", "prompt": [[1, 2, 3]] * rows, "n": n, "seed": 42,
            "max_tokens": 2048, "temperature": 1., "top_p": 1., "top_k": -1,
            "stop": ["END"], "include_stop_str_in_output": True, "return_token_ids": True}

    def response(self, part):
        return {"model": part["model"], "choices": [
            {"index": j, "token_ids": [part["seed"] + j], "text": "x"} for j in range(part["n"])],
            "usage": {"prompt_tokens": 3, "completion_tokens": part["n"], "total_tokens": 3 + part["n"]}}

    def test_512_distinct_child_streams_for_identical_prompts(self):
        parts = split_requests(self.request())
        seeds = [part["seed"] + j for part in parts for j in range(part["n"])]
        self.assertEqual(len(seeds), 512)
        self.assertEqual(len(set(seeds)), 512)
        self.assertEqual(seeds, list(range(42, 554)))

    def test_single_prompt_training_group_is_unchanged(self):
        request = self.request(rows=1)
        self.assertEqual(split_requests(request), [request])

    def test_stopping_distribution_and_pairing_are_preserved(self):
        request = self.request()
        original = copy.deepcopy(request)
        parts = split_requests(request)
        self.assertEqual(request, original)
        self.assertEqual(parts, split_requests(request))
        for part in parts:
            for key in ("model", "n", "max_tokens", "temperature", "top_p", "top_k", "stop", "include_stop_str_in_output"):
                self.assertEqual(part[key], request[key])

    def test_merge_has_exact_order_native_ids_and_usage(self):
        request = self.request(rows=2)
        parts = split_requests(request)
        result = merge_responses(request, [self.response(p) for p in parts])
        self.assertEqual([c["index"] for c in result["choices"]], list(range(16)))
        self.assertEqual([c["token_ids"][0] for c in result["choices"]], list(range(42, 58)))
        self.assertEqual(result["usage"]["completion_tokens"], 16)

    def test_incomplete_response_is_not_cached_as_complete(self):
        request = self.request(rows=1)
        response = self.response(request)
        response["choices"].pop()
        with self.assertRaises(ValueError):
            merge_responses(request, [response])

    def test_missing_seed_rejected(self):
        request = self.request()
        request["seed"] = None
        with self.assertRaises(ValueError):
            split_requests(request)

    def test_completed_parts_are_reused_without_more_sampling(self):
        request = self.request(rows=10)
        calls, lock = [], threading.Lock()
        def http(url, part, timeout):
            with lock:
                calls.append(part)
            return self.response(part)
        def write(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))
        with tempfile.TemporaryDirectory() as directory:
            parts_dir = Path(directory) / "parts"
            read = lambda path: json.loads(path.read_text())
            first = complete("http://fake", request, parts_dir, http, read, write)
            second = complete("http://fake", request, parts_dir, http, read, write)
            self.assertEqual(len(calls), 10)
            self.assertEqual(first["choices"], second["choices"])
            self.assertEqual(second["recovery_accounting"]["cached_prompt_requests"], 10)
            self.assertEqual(second["recovery_accounting"]["new_prompt_requests"], 0)
            changed = dict(request, model="another_policy")
            with self.assertRaises(RuntimeError):
                complete("http://fake", changed, parts_dir, http, read, write)
            self.assertEqual(len(calls), 10)

    def test_partial_recovery_samples_only_missing_prompts(self):
        request = self.request(rows=3)
        calls = []
        def write(path, obj):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = split_requests(request)[0]
            write(root / "prompt_0000.json", {"request": part, "response": self.response(part), "seconds": 2.})
            def http(url, part, timeout):
                calls.append(part)
                return self.response(part)
            result = complete("http://fake", request, root, http,
                              lambda path: json.loads(path.read_text()), write)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["recovery_accounting"]["cached_prompt_requests"], 1)
            validate_response(request, result)

    def test_legacy_response_and_misassigned_streams_fail_closed(self):
        request = self.request(rows=2)
        parts = split_requests(request)
        result = merge_responses(request, [self.response(p) for p in parts])
        validate_response(request, result)
        result["prompt_parent_seeds"] = [42, 42]
        with self.assertRaises(RuntimeError):
            validate_response(request, result)
        with self.assertRaises(RuntimeError):
            validate_response(request, self.response(request))
        records = [{"sampling_protocol": PROTOCOL, "sampling_parent_seed": 42+i//8*8,
            "sampling_child_seed": 42+i, "instance_index": i//8, "rollout_index": i%8,
            "prompt_token_ids": [1,2,3]} for i in range(16)]
        validate_records(request, records)
        records[-1]["sampling_child_seed"] = 42
        with self.assertRaises(RuntimeError):
            validate_records(request, records)

    def test_request_concurrency_is_bounded(self):
        with self.assertRaises(ValueError):
            complete("http://fake", self.request(), Path("unused"), None, None, None, max_workers=9)


if __name__ == "__main__":
    unittest.main()
