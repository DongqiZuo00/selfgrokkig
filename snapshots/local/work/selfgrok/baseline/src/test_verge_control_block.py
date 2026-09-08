"""Synthetic block/entrypoint integration; no Slurm submission or model calls."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from verge_control_block import validate_block, phases_for, validate_launch, prepared_round, validate_prepared
from verge_control_protocol import segment_config

SOURCE_EXP = Path(__file__).resolve().parents[1]


class ControlBlockTests(unittest.TestCase):
    def test_preexecution_real_plan_has_one_global_quarter_warmup(self):
        block = json.loads((SOURCE_EXP / "manifests/verge_control_v1_direct_protocol.json").read_text())
        validate_block(block)
        self.assertEqual(block["dense_warmup_loss_tokens"],sum(block["segment_loss_tokens"])//4)
        self.assertEqual(block["total_block_loss_tokens"],3*sum(block["segment_loss_tokens"]))
        phases = [phases_for(block,"dense_then_binary",r) for r in range(6)]
        self.assertEqual([p["reward_mode"] for p in phases[0]],["dense"])
        self.assertEqual([p["tokens"] for p in phases[1]],[262144,262144])
        self.assertTrue(all(p["reward_mode"] == "binary" for part in phases[2:] for p in part))

    def block(self):
        return {"protocol":"verge_control_v1_direct_block", "labels":["binary","dense","dense_then_binary"],
            "segment_loss_tokens":[524288]*6, "training_seed":42, "maximum_total_gpus":2,
            "maximum_total_cpu_memory_gb":64, "requires_primary_complete":True,
            "optimizer_reset":"fresh_each_segment", "total_compute_matched":False,
            "dense_warmup_loss_tokens":524288+16}

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def fixture(self, exp):
        baseline = json.loads((SOURCE_EXP / "manifests/verge_mistral_round1.json").read_text(encoding="utf-8"))
        self.write(exp / "manifests/verge_mistral_round1.json", baseline)
        block = self.block()
        self.write(exp / "manifests/verge_control_v1_direct_protocol.json", block)
        self.write(exp / "raw_results/verge_control_v1/protocol_frozen.json", block)
        self.write(exp / "raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json",
                   {"primary_block_complete":True, "outer_rounds":24})
        source = "checkpoints/verge_book_v2_initial_solver/resume_u0000"
        self.write(exp / source / "verge_committed.json", {"update":0})
        # Presence-only test placeholders are never loaded as model tensors.
        for name in ("adapter_model.safetensors", "state.pt"):
            (exp / source / name).write_bytes(b"synthetic fixture; not model weights")
        cfg = segment_config(exp, "dense_then_binary", 0,
            phases=phases_for(block,"dense_then_binary",0), starting_checkpoint=source,
            block_protocol="manifests/verge_control_v1_direct_protocol.json")
        self.write(exp / "manifests" / (cfg["protocol_version"]+".json"), cfg)
        return cfg

    def test_global_warmup_never_restarts_after_round_boundary(self):
        block = self.block()
        phases = [phases_for(block,"dense_then_binary",r) for r in range(6)]
        self.assertEqual([p["reward_mode"] for p in phases[0]], ["dense"])
        self.assertEqual([p["tokens"] for p in phases[1]], [16,524272])
        self.assertTrue(all(p["reward_mode"] == "binary" for part in phases[2:] for p in part))
        self.assertEqual(sum(p["tokens"] for part in phases for p in part if p["reward_mode"] == "dense"),524304)

    def test_missing_warmup_or_enlarged_block_rejected(self):
        for key, value in (("dense_warmup_loss_tokens", None), ("dense_warmup_loss_tokens",0),
                           ("maximum_total_gpus",3), ("total_compute_matched",True),
                           ("segment_loss_tokens",[524288]*7)):
            with self.assertRaises(ValueError):
                validate_block(dict(self.block(), **{key:value}))

    def test_launch_requires_separately_frozen_unchanged_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            cfg = self.fixture(exp)
            self.assertEqual(validate_launch(exp,cfg), self.block())
            path = exp / "raw_results/verge_control_v1/protocol_frozen.json"
            self.write(path, dict(self.block(), dense_warmup_loss_tokens=32))
            with self.assertRaises(RuntimeError):
                validate_launch(exp,cfg)
            path.unlink()
            with self.assertRaises(RuntimeError):
                validate_launch(exp,cfg)

    def test_active_primary_cannot_enable_new_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            cfg = self.fixture(exp)
            (exp / "raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json").unlink()
            with self.assertRaises(RuntimeError):
                validate_launch(exp,cfg)

    def test_config_drift_and_uncommitted_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            cfg = self.fixture(exp)
            with self.assertRaises(RuntimeError):
                validate_launch(exp,dict(cfg,scope_max_drop=.1))
            (exp / cfg["solver_start"] / "verge_committed.json").unlink()
            with self.assertRaises(RuntimeError):
                validate_launch(exp,cfg)

    def test_preparer_only_uses_declared_target_and_start(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.fixture(Path(temp))
            initial = {"checkpoint":cfg["solver_start"]}
            frozen = prepared_round(cfg,initial)
            self.assertIsNone(frozen["generation"])
            self.assertEqual(frozen["branches"][0]["stages"][0]["path"],cfg["target_train"])
            validate_prepared(cfg,frozen)
            altered = copy.deepcopy(frozen)
            altered["branches"][0]["id"] = "foreign_branch"
            with self.assertRaises(RuntimeError):
                validate_prepared(cfg,altered)

    def test_later_segment_requires_verified_exact_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            cfg0 = self.fixture(exp)
            source = "checkpoints/verge_control_v1_dense_then_binary_r00_direct/resume_u0030"
            self.write(exp / source / "verge_committed.json", {"update":30})
            for name in ("adapter_model.safetensors", "state.pt"):
                (exp / source / name).write_bytes(b"synthetic fixture")
            cfg1 = segment_config(exp,"dense_then_binary",1,
                phases=phases_for(self.block(),"dense_then_binary",1), starting_checkpoint=source,
                block_protocol=cfg0["control_block_protocol"])
            with self.assertRaises(RuntimeError):
                validate_launch(exp,cfg1)
            self.write(exp / "raw_results/verge_control_v1_dense_then_binary_r00/branches/0/complete.json",
                {"control_segment_verified":True, "control_label":"dense_then_binary",
                 "checkpoint":source, "training":{"train_tokens":524288}})
            validate_launch(exp,cfg1)

    def core_module(self, name):
        spec = importlib.util.spec_from_file_location("synthetic_core", SOURCE_EXP / "src/verge_round_core.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"VERGE_VERSION":name}):
            spec.loader.exec_module(module)
        return module

    def test_shared_entrypoint_routes_only_explicit_control_config(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            cfg = self.fixture(exp)
            core = self.core_module(cfg["protocol_version"])
            core.EXP, core.CFG_PATH = exp, exp / "manifests" / (cfg["protocol_version"]+".json")
            with patch.dict(os.environ, {"VERGE_CONTROL_SUITE":""}):
                with self.assertRaises(RuntimeError):
                    core.config()
            with patch.dict(os.environ, {"VERGE_CONTROL_SUITE":"verge_control_v1"}):
                self.assertEqual(core.config(), cfg)
                self.write(core.CFG_PATH, dict(cfg,protocol_version="verge_control_v1_binary_r00"))
                with self.assertRaises(RuntimeError):
                    core.config()
                core.CFG_PATH.unlink()
                with self.assertRaises(RuntimeError):
                    core.config()

    def test_primary_route_never_calls_control_authorizer(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            core = self.core_module("verge_book_v2_verge_r00")
            core.EXP, core.CFG_PATH = exp, exp / "manifests/verge_book_v2_verge_r00.json"
            cfg = {"book_suite":"verge_book_v2", "independent_prompt_streams":True}
            self.write(core.CFG_PATH,cfg)
            with patch.dict(os.environ,{"VERGE_BOOK_SUITE":"verge_book_v2"}), patch(
                    "verge_control_block.validate_launch", side_effect=AssertionError("Primary route changed")):
                self.assertEqual(core.config(),cfg)


if __name__ == "__main__":
    unittest.main()
