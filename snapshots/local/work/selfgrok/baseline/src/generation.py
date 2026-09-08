from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from common import (
    MODEL_ROOT,
    append_jsonl,
    apply_chat_template,
    completion_token_ids,
    load_config,
    read_jsonl,
    stable_int,
    verify_full,
)
from modeling import load_text_model, load_tokenizer


class InferenceEngine:
    def __init__(self, model: Any | None = None, tokenizer: Any | None = None):
        self.config = load_config()
        self.tokenizer = tokenizer or load_tokenizer("left")
        self.model = model or load_text_model()
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.sampling = self.config["sampling"]

    def _generate_chunk(
        self,
        tasks: list[tuple[int, int, dict[str, Any]]],
        stage_seed: int,
        chunk_index: int,
    ) -> list[dict[str, Any]]:
        prompts = [apply_chat_template(self.tokenizer, task[2]["messages"]) for task in tasks]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        if int(encoded.attention_mask.sum(1).max()) > int(self.sampling["max_prompt_tokens"]):
            raise RuntimeError("fixed max_prompt_tokens exceeded")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        torch.manual_seed(stable_int(stage_seed, chunk_index))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stable_int(stage_seed, chunk_index))
        start = time.time()
        with torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=float(self.sampling["temperature"]),
                top_p=float(self.sampling["top_p"]),
                top_k=int(self.sampling["top_k"]),
                max_new_tokens=int(self.sampling["max_completion_tokens"]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        elapsed = time.time() - start
        input_width = int(encoded["input_ids"].shape[1])
        output: list[dict[str, Any]] = []
        for task, sequence in zip(tasks, sequences):
            row_index, rollout_index, row = task
            token_ids = completion_token_ids(
                sequence,
                input_width,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            )
            completion = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            verdict = verify_full(completion, row["ground_truth"])
            output.append(
                {
                    "instance_index": row_index,
                    "instance_id": row["id"],
                    "rollout_index": rollout_index,
                    "completion": completion,
                    "completion_tokens": len(token_ids),
                    "completion_token_ids": token_ids,
                    "reward": verdict.reward,
                    "parse_valid": verdict.parse_valid,
                    "passed_cases": verdict.passed_cases,
                    "total_cases": verdict.total_cases,
                    "verifier_message": verdict.message,
                    "generation_seconds_share": elapsed / len(tasks),
                }
            )
        del encoded, sequences
        return output

    def score_rows(
        self,
        rows: list[dict[str, Any]],
        rollouts_per_instance: int,
        stage_seed: int,
        output_path: Path,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        tasks = [
            (row_index, rollout_index, row)
            for row_index, row in enumerate(rows)
            for rollout_index in range(rollouts_per_instance)
        ]
        completed: list[dict[str, Any]] = read_jsonl(output_path) if output_path.exists() else []
        expected_prefix = [
            (task[2]["id"], task[1]) for task in tasks[: len(completed)]
        ]
        actual_prefix = [
            (row["instance_id"], row["rollout_index"]) for row in completed
        ]
        if expected_prefix != actual_prefix:
            raise RuntimeError(f"non-contiguous or incompatible rollout resume file: {output_path}")
        for start in range(len(completed), len(tasks), batch_size):
            chunk_index = start // batch_size
            chunk = tasks[start : start + batch_size]
            result = self._generate_chunk(chunk, stage_seed, chunk_index)
            append_jsonl(output_path, result)
            completed.extend(result)
            print(
                f"{output_path.name}: {len(completed)}/{len(tasks)} "
                f"success={sum(x['reward'] for x in completed)}",
                flush=True,
            )
        return completed


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [int(row["reward"]) for row in rows]
    parse_valid = [int(row["parse_valid"]) for row in rows]
    lengths = [int(row["completion_tokens"]) for row in rows]
    by_instance: dict[str, list[int]] = {}
    for row in rows:
        by_instance.setdefault(row["instance_id"], []).append(int(row["reward"]))
    return {
        "rollouts": len(rows),
        "full_pass_count": sum(rewards),
        "full_pass_rate": sum(rewards) / len(rewards),
        "parse_valid_count": sum(parse_valid),
        "parse_valid_rate": sum(parse_valid) / len(parse_valid),
        "mean_completion_tokens": sum(lengths) / len(lengths),
        "instance_success_rates": {
            instance_id: sum(values) / len(values) for instance_id, values in by_instance.items()
        },
    }
