"""GPU utilities shared by the isolated VERGE pilot, without legacy state mutation."""
from __future__ import annotations

import gc
import os
import subprocess
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_from_disk

from common import EXP_ROOT, atomic_json, read_json, stable_int
from generation import summarize_rollouts
from train_branch import save_checkpoint, prune_checkpoints
from verge_round_core import ROOT, VERSION, config, profile
from vllm_runtime import (VLLMServerPool as LegacyPool, VLLMRolloutClient as LegacyClient,
                          VLLM_PYTHON, wait_for_server)
from common import MODEL_ROOT


class VLLMRolloutClient(LegacyClient):
    def __init__(self, url, gpu=0):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        identity = visible[gpu] if visible and visible[0] else gpu
        super().__init__(url, identity)

    def score_rows(self, rows, rollouts_per_instance, stage_seed, output_path,
                   telemetry_path=None, verification_case_sets=None, sampling_seed_key=None):
        if config().get("repair_integrated"):
            if verification_case_sets is not None:
                raise ValueError("Repair protocol never weakens a task's verifier")
            from verge_repair_sampling import score_rows
            return score_rows(self, rows, rollouts_per_instance, stage_seed, output_path,
                              telemetry_path, sampling_seed_key,
                              independent_prompt_streams=config().get("independent_prompt_streams", False))
        return super().score_rows(rows, rollouts_per_instance, stage_seed, output_path,
                                  telemetry_path, verification_case_sets, sampling_seed_key)


class VLLMServerPool(LegacyPool):
    """Job-isolated logs and Slurm-visible GPU mapping; legacy users stay untouched."""
    def start(self):
        if self.processes:
            return self.urls
        assert (MODEL_ROOT / "consolidated.safetensors").exists()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        log_root = ROOT / "service_logs" / (os.environ.get("SLURM_JOB_ID", "manual") + "_" +
                   os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        log_root.mkdir(parents=True, exist_ok=True)
        for gpu, url in zip(self.gpus, self.urls):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible[gpu] if visible and visible[0] else str(gpu)
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            if config().get("repair_integrated"):
                env["TVM_FFI_CACHE_DIR"] = str(EXP_ROOT.parents[1] / "caches/tvm-ffi")
            stdout = (log_root / f"gpu{gpu}.out").open("a", encoding="utf-8")
            stderr = (log_root / f"gpu{gpu}.err").open("a", encoding="utf-8")
            command = [str(VLLM_PYTHON), "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(MODEL_ROOT), "--served-model-name", "base", "--dtype", "bfloat16",
                "--attention-backend", "FLASH_ATTN", "--tensor-parallel-size", "1", "--language-model-only",
                "--max-model-len", "12288", "--max-num-seqs", "128", "--max-num-batched-tokens", "49152",
                "--max-cudagraph-capture-size", "128", "--gpu-memory-utilization", "0.40",
                "--no-enable-prefix-caching", "--enable-lora", "--max-lora-rank", "64", "--max-loras", "2",
                "--no-enable-log-requests", "--port", str(self.base_port + gpu)]
            if config().get("repair_integrated"):
                command += ["--tokenizer-mode", "hf", "--generation-config", "vllm",
                            "--logprobs-mode", "processed_logprobs",
                            "--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":false}']
            self.processes.append(subprocess.Popen(command, cwd=str(EXP_ROOT.parents[1]), env=env,
                                                   stdout=stdout, stderr=stderr))
            self.handles.extend([stdout, stderr])
            print(f"vLLM allocation-visible GPU {env['CUDA_VISIBLE_DEVICES']} starting at {url}", flush=True)
        for url, process in zip(self.urls, self.processes):
            wait_for_server(url, process)
            print(f"vLLM healthy {url}", flush=True)
        return self.urls


def rows(path):
    return [dict(r) for r in load_from_disk(str(EXP_ROOT / path))]


def optimizer_for(model, lr):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, betas=(.9, .95), weight_decay=0.0)
    return optimizer, LambdaLR(optimizer, lambda _: 1.0)


def commit(model, optimizer, scheduler, branch, update, state):
    path = save_checkpoint(model, optimizer, scheduler, branch, update, state)
    atomic_json(path / "verge_committed.json", {"update": update, "time": time.time()})
    prune_checkpoints(branch, {0, update}, resumable_to_keep=2)
    return path


def latest(branch):
    candidates = list((EXP_ROOT / "checkpoints" / branch).glob("resume_u*/verge_committed.json"))
    if not candidates:
        return None
    marker = max(candidates, key=lambda p: int(p.parent.name.removeprefix("resume_u")))
    return marker.parent


def free_models():
    gc.collect()
    torch.cuda.empty_cache()


def endpoint(client, output_dir, *, checkpoint, selection, scope):
    cfg = config()
    stream = cfg.get("random_stream_id", VERSION)
    target_scored = client.score_rows(
        selection, cfg["selection_samples_per_instance"],
        stable_int(stream, 42, "selection"),
        output_dir / "target.jsonl", sampling_seed_key=f"{stream}_selection",
    )
    target = profile(target_scored, selection)
    scope_scored = client.score_rows(
        scope, cfg["scope_samples"], stable_int(stream, 42, "scope"),
        output_dir / "scope.jsonl", sampling_seed_key=f"{stream}_scope",
    )
    scope_summary = summarize_rollouts(scope_scored)
    scope_summary["mean_case_fraction"] = sum(
        r["passed_cases"] / max(1, r["total_cases"]) for r in scope_scored
    ) / len(scope_scored)
    result = {"checkpoint": checkpoint, "target": target, "scope": scope_summary}
    if cfg.get("repair_integrated"):
        result["interface_version"] = cfg["output_interface"]
    if cfg.get("independent_prompt_streams"):
        from verge_independent_sampling import PROTOCOL
        result["sampling_protocol"] = PROTOCOL
        result["configured_child_streams"] = {"target": len(target_scored), "scope": len(scope_scored)}
    atomic_json(output_dir / "endpoint.json", result)
    return result
