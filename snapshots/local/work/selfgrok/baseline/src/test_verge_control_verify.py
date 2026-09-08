"""Synthetic on-disk segment fixtures; not training or endpoint model samples."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from common import group_advantages
from verge_control_protocol import segment_config
from verge_control_verify import validate_segment
from verge_independent_sampling import PROTOCOL


class ControlVerificationTests(unittest.TestCase):
    def fixture(self, root):
        cfg = segment_config(Path(__file__).resolve().parents[1], "dense", 0,
            phases=[{"kind":"target", "reward_mode":"dense", "tokens":8}],
            starting_checkpoint="checkpoints/verge_book_v2_initial_solver/resume_u0000",
            block_protocol="manifests/verge_control_v1_direct_protocol.json")
        frozen = {"config":cfg, "branches":[{"index":0, "stages":cfg["control_phases"]}]}
        records = [{"instance_id":"synthetic", "instance_index":0, "rollout_index":i,
            "reward":0, "passed_cases":i%4, "total_cases":4,
            "optimization_reward":(i%4)/4, "optimization_reward_mode":"dense",
            "prompt_role":"target", "sampling_protocol":PROTOCOL,
            "sampling_parent_seed":100, "sampling_child_seed":100+i,
            "prompt_token_ids":[1], "completion_token_ids":[2], "completion_tokens":1,
            "max_tokens":2048} for i in range(8)]
        self.write_rows(root / "train/update_0001.jsonl", records)
        self.write_json(root / "train/update_0001_allocation.json", {
            "stage":0, "advantage_source":"optimization_reward", "loss_masks":[[1]]*8,
            "advantages":group_advantages([r["optimization_reward"] for r in records], 8)})
        self.write_json(root / "train/update_0001_summary.json", {"optimizer_step":True})
        state = {"update":1, "used_by_stage":[8], "train_tokens":8,
                 "generated_tokens":8, "nonzero_advantage_tokens":8,
                 "optimizer_steps":1, "target_successes":0}
        for name, prompts, n in (("target",64,8), ("scope",32,4)):
            request = {"prompt":[[1]]*prompts, "n":n, "seed":1000}
            response = {"backend_protocol":PROTOCOL,
                "prompt_parent_seeds":[1000+i*n for i in range(prompts)],
                "child_seed_count":prompts*n, "choices":[{"index":i} for i in range(prompts*n)]}
            rows = [{"instance_index":i//n, "rollout_index":i%n, "prompt_token_ids":[1],
                "sampling_protocol":PROTOCOL, "sampling_parent_seed":1000+(i//n)*n,
                "sampling_child_seed":1000+i} for i in range(prompts*n)]
            self.write_json(root / f"{name}.response.json", {
                "backend_protocol":PROTOCOL, "request":request, "response":response})
            self.write_rows(root / f"{name}.jsonl", rows)
        return frozen, {"training":state}

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")

    def test_dense_learning_never_becomes_a_full_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, result = self.fixture(root)
            verified = validate_segment(frozen, result, root)
            self.assertTrue(verified["control_segment_verified"])
            self.assertFalse(verified["comparison_complete"])
            self.assertEqual(result["training"]["target_successes"], 0)

    def test_partial_rewards_cannot_be_counted_as_full_successes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, result = self.fixture(root)
            result["training"]["target_successes"] = 6
            with self.assertRaises(AssertionError):
                validate_segment(frozen, result, root)

    def test_wrong_advantages_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, result = self.fixture(root)
            path = root / "train/update_0001_allocation.json"
            saved = json.loads(path.read_text())
            saved["advantages"] = [0.]*8
            self.write_json(path, saved)
            with self.assertRaises(AssertionError):
                validate_segment(frozen, result, root)

    def test_reused_training_child_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, result = self.fixture(root)
            path = root / "train/update_0001.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[1]["sampling_child_seed"] = records[0]["sampling_child_seed"]
            self.write_rows(path, records)
            with self.assertRaises(RuntimeError):
                validate_segment(frozen, result, root)

    def test_reused_endpoint_child_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen, result = self.fixture(root)
            path = root / "target.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[8]["sampling_child_seed"] = records[0]["sampling_child_seed"]
            self.write_rows(path, records)
            with self.assertRaises(RuntimeError):
                validate_segment(frozen, result, root)

    def test_budget_and_full_group_mask_accounting_are_enforced(self):
        for key, bad in (("train_tokens",9), ("generated_tokens",9), ("optimizer_steps",0)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                frozen, result = self.fixture(root)
                result["training"][key] = bad
                with self.assertRaises(AssertionError):
                    validate_segment(frozen, result, root)


if __name__ == "__main__":
    unittest.main()
