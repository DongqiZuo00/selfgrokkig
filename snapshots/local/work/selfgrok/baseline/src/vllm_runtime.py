from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


from common import (
    EXP_ROOT,
    MODEL_ROOT,
    append_jsonl,
    apply_chat_template,
    extract_program,
    load_config,
    read_jsonl,
    stable_int,
    verify_all_tests,
)
from modeling import load_tokenizer


VLLM_PYTHON = EXP_ROOT.parents[1] / "envs" / "vllm" / "bin" / "python"
PROGRAM_STOPS = ["\nEND end\n```", "\nEND end\r\n```"]


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"text": body.decode("utf-8", errors="replace")}



def wait_for_server(base_url: str, process: subprocess.Popen, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}: {base_url}")
        try:
            request = urllib.request.Request(base_url + "/health")
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM server did not become healthy: {base_url}: {last_error}")


class VLLMServerPool:
    def __init__(self, gpus: tuple[int, ...] = (0, 1), base_port: int = 8100):
        self.gpus = gpus
        self.base_port = base_port
        self.processes: list[subprocess.Popen] = []
        self.handles: list[Any] = []
        self.urls = [f"http://127.0.0.1:{base_port + gpu}" for gpu in gpus]

    def start(self) -> list[str]:
        if self.processes:
            return self.urls
        native_checkpoint = MODEL_ROOT / "consolidated.safetensors"
        if "mistral" in str(MODEL_ROOT).lower() and not native_checkpoint.exists():
            raise RuntimeError(
                "Mistral vLLM startup requires consolidated.safetensors; "
                "run download_mistral_model.py before submitting GPU work"
            )
        log_root = EXP_ROOT / "logs" / "vllm"
        log_root.mkdir(parents=True, exist_ok=True)
        for gpu, url in zip(self.gpus, self.urls):
            stdout = (log_root / f"gpu{gpu}.out").open("a", encoding="utf-8")
            stderr = (log_root / f"gpu{gpu}.err").open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            env["FLASHINFER_WORKSPACE_BASE"] = str(EXP_ROOT.parents[1] / "caches" / "flashinfer_workspace")
            env["TOKENIZERS_PARALLELISM"] = "false"
            command = [
                str(VLLM_PYTHON),
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(MODEL_ROOT),
                "--served-model-name",
                "base",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "FLASH_ATTN",
                "--tensor-parallel-size",
                "1",
                "--language-model-only",
                "--max-model-len",
                "12288",
                "--max-num-seqs",
                "128",
                "--max-num-batched-tokens",
                "49152",
                "--max-cudagraph-capture-size",
                "128",
                "--gpu-memory-utilization",
                "0.40",
                "--no-enable-prefix-caching",
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--max-loras",
                "2",
                "--no-enable-log-requests",
                "--port",
                str(self.base_port + gpu),
            ]
            process = subprocess.Popen(
                command,
                cwd=str(EXP_ROOT.parents[1]),
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            self.processes.append(process)
            self.handles.extend([stdout, stderr])
            print(f"vLLM GPU {gpu} starting at {url}", flush=True)
        for url, process in zip(self.urls, self.processes):
            wait_for_server(url, process)
            print(f"vLLM healthy: {url}", flush=True)
        return self.urls

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 30
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in self.handles:
            handle.close()
        self.processes.clear()
        self.handles.clear()

    def __enter__(self) -> list[str]:
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def _verify_payload(payload: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    completion, tests = payload
    started = time.perf_counter()
    verdict = verify_all_tests(completion, tests)
    return {
        "reward": verdict.reward,
        "parse_valid": verdict.parse_valid,
        "passed_cases": verdict.passed_cases,
        "total_cases": verdict.total_cases,
        "verifier_message": verdict.message,
        "verifier_seconds": time.perf_counter() - started,
    }


def _metric_value(text: str, fragment: str) -> float:
    values = []
    for line in text.splitlines():
        if fragment in line and not line.startswith("#"):
            try:
                values.append(float(line.rsplit(" ", 1)[-1]))
            except ValueError:
                pass
    return sum(values)


class TelemetrySampler:
    def __init__(self, base_url: str, gpu: int, output_path: Path):
        self.base_url = base_url
        self.gpu = gpu
        self.output_path = output_path
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"time": time.time(), "gpu": self.gpu}
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu}",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            utilization, memory, power = [float(value.strip()) for value in query.split(",")]
            sample.update(
                {"gpu_utilization": utilization, "gpu_memory_mib": memory, "gpu_power_w": power}
            )
        except Exception as exc:
            sample["gpu_error"] = str(exc)
        try:
            with urllib.request.urlopen(self.base_url + "/metrics", timeout=2) as response:
                metrics = response.read().decode("utf-8")
            sample["active_sequences"] = _metric_value(metrics, "num_requests_running")
            sample["waiting_sequences"] = _metric_value(metrics, "num_requests_waiting")
        except Exception as exc:
            sample["metrics_error"] = str(exc)
        return sample

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(self._sample())
            self.stop_event.wait(2)

    def __enter__(self) -> "TelemetrySampler":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        append_jsonl(self.output_path, self.samples)


class VLLMRolloutClient:
    def __init__(self, base_url: str, gpu: int, verifier_workers: int = 8):
        self.base_url = base_url.rstrip("/")
        self.gpu = gpu
        self.verifier_workers = verifier_workers
        self.config = load_config()
        self.sampling = self.config["sampling"]
        self.tokenizer = load_tokenizer("left")
        self.model_name = "base"
        self.loaded_lora: str | None = None

    def load_lora(self, name: str, path: Path) -> None:
        started = time.perf_counter()
        http_json(
            self.base_url + "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": str(path.resolve())},
            timeout=300,
        )
        previous = self.loaded_lora
        self.loaded_lora = name
        self.model_name = name
        if previous and previous != name:
            http_json(
                self.base_url + "/v1/unload_lora_adapter",
                {"lora_name": previous},
                timeout=300,
            )
        print(f"LoRA sync {name}: {time.perf_counter() - started:.3f}s", flush=True)

    def use_base(self) -> None:
        if self.loaded_lora:
            http_json(
                self.base_url + "/v1/unload_lora_adapter",
                {"lora_name": self.loaded_lora},
                timeout=300,
            )
        self.loaded_lora = None
        self.model_name = "base"

    def score_rows(
        self,
        rows: list[dict[str, Any]],
        rollouts_per_instance: int,
        stage_seed: int,
        output_path: Path,
        telemetry_path: Path | None = None,
        verification_case_sets: list[list[list[dict[str, Any]]]] | None = None,
        sampling_seed_key: str | None = None,
    ) -> list[dict[str, Any]]:
        expected = len(rows) * rollouts_per_instance
        if verification_case_sets is not None:
            if len(verification_case_sets) != len(rows) or any(
                len(case_sets) != rollouts_per_instance
                for case_sets in verification_case_sets
            ):
                raise RuntimeError("verification_case_sets does not match rollout shape")
        if output_path.exists():
            existing = read_jsonl(output_path)
            if len(existing) == expected:
                return existing
            if existing:
                raise RuntimeError(f"partial atomic rollout unit requires quarantine: {output_path}")
        prompts = [apply_chat_template(self.tokenizer, row["messages"]) for row in rows]
        request_payload = {
            "model": self.model_name,
            "prompt": prompts,
            "n": rollouts_per_instance,
            "max_tokens": int(
                os.environ.get(
                    "ROLLOUT_MAX_COMPLETION_TOKENS",
                    self.sampling["max_completion_tokens"],
                )
            ),
            "temperature": float(self.sampling["temperature"]),
            "top_p": float(self.sampling["top_p"]),
            "top_k": int(self.sampling["top_k"]),
            # Adapter names are branch-specific. A shared key keeps matched
            # branches on the same sampling stream while preserving the legacy
            # default for older protocols.
            "seed": stable_int(stage_seed, sampling_seed_key or self.model_name),
            "stop": PROGRAM_STOPS,
            "include_stop_str_in_output": True,
        }
        telemetry_path = telemetry_path or output_path.with_suffix(".telemetry.jsonl")
        started = time.perf_counter()
        with TelemetrySampler(self.base_url, self.gpu, telemetry_path):
            response = http_json(
                self.base_url + "/v1/completions",
                request_payload,
                timeout=24 * 3600,
            )
        generation_seconds = time.perf_counter() - started
        choices = sorted(response["choices"], key=lambda item: int(item["index"]))
        if len(choices) != expected:
            raise RuntimeError(f"vLLM returned {len(choices)} choices, expected {expected}")
        verification_inputs: list[tuple[str, list[dict[str, Any]]]] = []
        provisional: list[dict[str, Any]] = []
        for flat_index, choice in enumerate(choices):
            row_index = flat_index // rollouts_per_instance
            rollout_index = flat_index % rollouts_per_instance
            row = rows[row_index]
            completion = str(choice["text"])
            token_ids = self.tokenizer(completion, add_special_tokens=False).input_ids
            provisional.append(
                {
                    "instance_index": row_index,
                    "instance_id": row["id"],
                    "rollout_index": rollout_index,
                    "completion": completion,
                    "program": extract_program(completion),
                    "completion_tokens": len(token_ids),
                    "completion_token_ids": [int(value) for value in token_ids],
                    "finish_reason": choice.get("finish_reason"),
                    "stop_reason": choice.get("stop_reason"),
                    "generation_seconds_share": generation_seconds / expected,
                    "request_wall_seconds": generation_seconds,
                }
            )
            cases = (
                row["ground_truth"]
                if verification_case_sets is None
                else verification_case_sets[row_index][rollout_index]
            )
            verification_inputs.append((completion, cases))
        verify_started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=self.verifier_workers) as executor:
            verdicts = list(executor.map(_verify_payload, verification_inputs, chunksize=8))
        verifier_wall = time.perf_counter() - verify_started
        completed = []
        for item, verdict in zip(provisional, verdicts):
            completed.append({**item, **verdict, "verifier_wall_seconds": verifier_wall})
        append_jsonl(output_path, completed)
        output_tokens = sum(item["completion_tokens"] for item in completed)
        print(
            f"{output_path.name}: {expected} rollouts, {generation_seconds / expected:.4f}s/rollout, "
            f"{output_tokens / generation_seconds:.1f} output tok/s, verify={verifier_wall:.2f}s",
            flush=True,
        )
        return completed
