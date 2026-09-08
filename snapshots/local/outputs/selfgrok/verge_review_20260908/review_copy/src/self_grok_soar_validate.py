from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, read_json, stable_int
from generation import summarize_rollouts
from self_grok_soar_loop import decision_path, protocol, validation_path
from vllm_runtime import VLLMRolloutClient, VLLMServerPool


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **summarize_rollouts(rows),
        "mean_case_fraction": sum(
            float(item["passed_cases"]) / max(1, int(item["total_cases"]))
            for item in rows
        )
        / max(1, len(rows)),
    }


def evaluate(
    client: VLLMRolloutClient,
    branch_id: str,
    rows: list[dict[str, Any]],
    samples: int,
    outer: int,
    split: str,
    raw_path: Path,
) -> dict[str, Any]:
    scored = client.score_rows(
        rows,
        samples,
        stable_int("soar_held_out_validation", outer, split),
        raw_path,
        sampling_seed_key=f"soar_o{outer:03d}_held_out_{split}",
    )
    return {
        "branch_id": branch_id,
        "split": split,
        "instances": len(rows),
        "samples_per_instance": samples,
        **summarize(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    role = "direct" if args.index == 0 else "promoted"
    output = validation_path(args.outer, role)
    if output.exists():
        print(json.dumps(read_json(output), indent=2), flush=True)
        return
    decision = read_json(decision_path(args.outer))
    branch_id = (
        str(decision["direct_branch_id"])
        if role == "direct"
        else str(decision["promotion"]["branch_id"])
    )
    cfg = protocol()
    goal = int(cfg["inner_loop"]["updates"])
    checkpoint = EXP_ROOT / "checkpoints" / branch_id / f"resume_u{goal:04d}"
    if not (checkpoint / "adapter_config.json").exists():
        raise RuntimeError(f"missing validation checkpoint: {checkpoint}")
    development = [
        dict(row) for row in load_from_disk(str(DATA_ROOT / "target_splits" / "development"))
    ]
    scope_dataset = load_from_disk(str(DATA_ROOT / "root" / "basic_mix"))
    scope = [
        dict(scope_dataset[index])
        for index in range(int(cfg["inner_loop"]["scope_instances"]))
    ]
    raw_root = output.parent / role
    raw_root.mkdir(parents=True, exist_ok=True)
    with VLLMServerPool(gpus=(0,), base_port=args.port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora(f"soar_validate_o{args.outer:03d}_{role}", checkpoint)
        result = {
            "outer": args.outer,
            "role": role,
            "branch_id": branch_id,
            "development": evaluate(
                client,
                branch_id,
                development,
                16,
                args.outer,
                "development",
                raw_root / "development.jsonl",
            ),
            "scope": evaluate(
                client,
                branch_id,
                scope,
                int(cfg["inner_loop"]["scope_rollouts_per_instance"]),
                args.outer,
                "scope",
                raw_root / "scope.jsonl",
            ),
        }
        client.use_base()
    atomic_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
