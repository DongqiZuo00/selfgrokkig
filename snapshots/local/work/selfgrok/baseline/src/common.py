from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = WORK_ROOT / "vendor"
MODEL_REPO_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
MODEL_ROOT = WORK_ROOT / "models" / "Ministral-3-3B-Instruct-2512-BF16"
if "qwen" in str(MODEL_ROOT).lower() or "qwen" in MODEL_REPO_ID.lower():
    raise RuntimeError("Qwen backbones are prohibited for this experiment")
DATA_ROOT = EXP_ROOT / "data"
CONFIG_PATH = EXP_ROOT / "configs" / "experiment.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs() -> None:
    for name in (
        "configs",
        "manifests",
        "scripts",
        "src",
        "logs",
        "checkpoints",
        "raw_results",
        "aggregated_results",
        "figures",
        "paper_outputs",
    ):
        (EXP_ROOT / name).mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    text = "|".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % modulo


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def uncertainty_rzero(p: float) -> float:
    """R-Zero paper uncertainty: maximized at verified success rate one half."""
    return float(1.0 - 2.0 * abs(float(p) - 0.5))


def uncertainty_matched(p: float) -> float:
    """Frozen fast-protocol uncertainty, U = 4 p (1-p)."""
    return float(4.0 * float(p) * (1.0 - float(p)))


def group_advantages(rewards: list[int], group_size: int = 8) -> list[float]:
    result: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = np.asarray(rewards[start : start + group_size], dtype=np.float32)
        if len(group) != group_size:
            raise ValueError("incomplete GRPO group")
        std = float(group.std())
        if std < 1e-6:
            result.extend([0.0] * group_size)
        else:
            result.extend(((group - float(group.mean())) / (std + 1e-6)).tolist())
    return result


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    arguments = {"tokenize": False, "add_generation_prompt": True}
    if tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **arguments,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **arguments)

    # Ministral 3 ships its official Tekken control tokens but deliberately
    # leaves the generic HF chat_template unset.  Reproduce the native Mistral
    # instruction framing for the system/user/assistant roles used here.
    parts = [tokenizer.bos_token or "<s>"]
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        if role == "system":
            parts.extend(["[SYSTEM_PROMPT]", content, "[/SYSTEM_PROMPT]"])
        elif role == "user":
            parts.extend(["[INST]", content, "[/INST]"])
        elif role == "assistant":
            parts.extend([content, tokenizer.eos_token or "</s>"])
        else:
            raise ValueError(f"unsupported Mistral chat role: {role}")
    return "".join(parts)


def completion_token_ids(sequence: torch.Tensor, input_width: int, eos_id: int, pad_id: int) -> list[int]:
    values = sequence[input_width:].tolist()
    result: list[int] = []
    for token in values:
        if token == pad_id:
            break
        result.append(int(token))
        if token == eos_id:
            break
    return result


def _load_official_verifier():
    verifier_parent = VENDOR_ROOT / "rl-grok-recipe"
    if str(verifier_parent) not in sys.path:
        sys.path.insert(0, str(verifier_parent))
    from manufactoria.verifier.manufactoria_parser import create_robot_factory

    return create_robot_factory


CREATE_ROBOT_FACTORY = _load_official_verifier()


@dataclass
class Verification:
    reward: int
    parse_valid: bool
    passed_cases: int
    total_cases: int
    message: str = ""


def extract_program(completion: str) -> str:
    """Extract the first complete Manufactoria block without altering raw output."""
    fenced = re.search(
        r"```(?:manufactoria)?\s*\n(.*?^END\s+end\s*$).*?```",
        completion,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if fenced:
        return fenced.group(1).strip()
    start = re.search(r"^START\s+start:\s*$", completion, flags=re.MULTILINE)
    if start:
        end = re.search(
            r"^END\s+end\s*$",
            completion[start.start() :],
            flags=re.MULTILINE,
        )
        if end:
            return completion[start.start() : start.start() + end.end()].strip()
    return completion.strip()


def verify_full(completion: str, test_cases: list[dict[str, Any]]) -> Verification:
    """Exact synchronous equivalent of the upstream full verifier endpoint."""
    program = extract_program(completion)
    try:
        factory = CREATE_ROBOT_FACTORY(program)
    except Exception as exc:
        return Verification(0, False, 0, len(test_cases), str(exc))
    passed_count = 0
    try:
        for case in test_cases:
            result = factory.process_robot(case.get("input", ""))
            if case.get("check_output", True):
                expected = case.get("expected_output", "")
                has_regex = any(char in expected for char in [".", "+", "*", "?", "|", "(", ")"])
                if has_regex:
                    try:
                        output_matches = bool(re.fullmatch(expected, result.final_tape))
                    except re.error:
                        output_matches = result.final_tape == expected
                else:
                    output_matches = result.final_tape == expected
                passed = (output_matches and result.finished) == bool(case.get("expected_accepted", True))
            else:
                passed = result.finished == bool(case.get("expected_accepted", True))
            if not passed:
                return Verification(0, True, passed_count, len(test_cases), "first failing test")
            passed_count += 1
        reward = int(passed_count == len(test_cases))
        return Verification(reward, True, passed_count, len(test_cases))
    except Exception as exc:
        return Verification(0, True, passed_count, len(test_cases), str(exc))


def verify_all_tests(
    completion: str,
    test_cases: list[dict[str, Any]],
) -> Verification:
    """Evaluate every test while preserving the strict all-pass reward.

    Unlike verify_full, this does not stop at the first failure. The resulting
    passed_cases is therefore a valid dense signal, while reward remains the
    original binary full-pass verdict.
    """
    program = extract_program(completion)
    try:
        factory = CREATE_ROBOT_FACTORY(program)
    except Exception as exc:
        return Verification(0, False, 0, len(test_cases), str(exc))
    passed_count = 0
    errors: list[str] = []
    for index, case in enumerate(test_cases):
        try:
            result = factory.process_robot(case.get("input", ""))
            if case.get("check_output", True):
                expected = case.get("expected_output", "")
                has_regex = any(
                    char in expected for char in [".", "+", "*", "?", "|", "(", ")"]
                )
                if has_regex:
                    try:
                        output_matches = bool(re.fullmatch(expected, result.final_tape))
                    except re.error:
                        output_matches = result.final_tape == expected
                else:
                    output_matches = result.final_tape == expected
                passed = (output_matches and result.finished) == bool(
                    case.get("expected_accepted", True)
                )
            else:
                passed = result.finished == bool(
                    case.get("expected_accepted", True)
                )
            passed_count += int(passed)
        except Exception as exc:
            errors.append(f"test {index}: {exc}")
    reward = int(passed_count == len(test_cases))
    message = "; ".join(errors[:3])
    if not reward and not message:
        message = f"failed {len(test_cases) - passed_count} tests"
    return Verification(reward, True, passed_count, len(test_cases), message)


def tokenizer_prompt_lengths(tokenizer: Any, rows: list[dict[str, Any]]) -> list[int]:
    return [len(tokenizer(apply_chat_template(tokenizer, row["messages"])).input_ids) for row in rows]


def branch_id(kind: str, seed: int, candidate_id: str | None = None) -> str:
    if kind == "direct":
        return f"direct__seed{seed}"
    if not candidate_id:
        raise ValueError("candidate_id required")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id)
    return f"candidate__{safe}__seed{seed}"


def package_versions() -> dict[str, str]:
    import datasets
    import peft
    import transformers
    import trl

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
    }
