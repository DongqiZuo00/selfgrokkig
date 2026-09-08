from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_from_disk

from common import (
    DATA_ROOT,
    EXP_ROOT,
    MODEL_ROOT,
    VENDOR_ROOT,
    apply_chat_template,
    atomic_json,
    ensure_dirs,
    load_config,
    read_json,
    seed_everything,
    stable_int,
    write_jsonl,
)
from modeling import load_tokenizer


MANU_ROOT = VENDOR_ROOT / "rl-grok-recipe" / "manufactoria"
if str(MANU_ROOT) not in sys.path:
    sys.path.insert(0, str(MANU_ROOT))

from hf_file_wrapper import TrainingFileWrapper
from manufactoria_problem_generators import GeneratorConfig, GeneratorRegistry


SEQUENCE_ATTRIBUTES = {
    "append_sequence": "APPEND_SEQUENCE_LENGTH",
    "contains_ordered": "CONTAINS_ORDERED_LENGTH",
    "contains_substring": "CONTAINS_SUBSTRING_LENGTH",
    "ends_with": "ENDS_WITH_SEQ_LENGTH",
    "exact_sequence": "EXACT_SEQUENCE_LENGTH",
    "starts_with": "STARTS_WITH_SEQ_LENGTH",
    "prepend_sequence": "PREPEND_SEQUENCE_LENGTH",
}
NUMERIC_FAMILIES = {"numerical_comparison", "numerical_max_min", "numerical_operations"}


@contextmanager
def generator_overrides(family: str, config: dict[str, Any]):
    updates: dict[str, Any] = {
        "COUNT_THRESHOLDS": tuple(config["count_thresholds"]),
        "NUMERICAL_THRESHOLDS": tuple(config["numerical_thresholds"]),
        "REGEX_MAX_PATTERN_LENGTH": tuple(config["regex_max_pattern_length"]),
    }
    if family in SEQUENCE_ATTRIBUTES:
        updates[SEQUENCE_ATTRIBUTES[family]] = tuple(config["sequence_lengths"])
    if family == "prepend_sequence":
        updates["PREPEND_ENABLE_MUTATIONS"] = bool(
            config.get("prepend_enable_mutations", True)
        )
        updates["PREPEND_PATTERN_MUTATIONS"] = bool(config.get("prepend_pattern_mutations", True))
    old: dict[str, Any] = {}
    try:
        for key, value in updates.items():
            old[key] = getattr(GeneratorConfig, key)
            setattr(GeneratorConfig, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(GeneratorConfig, key, value)


def effective_color_mode(family: str, requested: str) -> str:
    if family in NUMERIC_FAMILIES:
        return "two_color"
    if family == "regex_same_num":
        return "four_color"
    return requested


def _params_for_distribution(
    family: str, candidate_config: dict[str, Any], requested_count: int
) -> tuple[list[dict[str, Any]], str]:
    generator = GeneratorRegistry.get_generator(family)
    mode = effective_color_mode(family, candidate_config["color_mode"])
    chars = GeneratorConfig.TWO_COLOR_CHARS if mode == "two_color" else GeneratorConfig.FOUR_COLOR_CHARS
    with generator_overrides(family, candidate_config):
        params = generator.generate_parameters(chars, requested_count)
    if not params:
        raise RuntimeError(f"official generator returned no parameters for {family}")
    return params, mode


def generate_distribution(
    family: str,
    candidate_config: dict[str, Any],
    count: int,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    """Generate instances with the upstream generator and prompt wrapper."""
    generator = GeneratorRegistry.get_generator(family)
    wrapper = TrainingFileWrapper()
    params, mode = _params_for_distribution(family, candidate_config, count)
    chars = GeneratorConfig.TWO_COLOR_CHARS if mode == "two_color" else GeneratorConfig.FOUR_COLOR_CHARS
    rows: list[dict[str, Any]] = []
    with generator_overrides(family, candidate_config):
        for index in range(count):
            item_seed = stable_int(seed, family, candidate_config["name"], split_name, index)
            seed_everything(item_seed)
            parameter = dict(params[index % len(params)])
            problem = generator.generate_problem(
                chars,
                is_four_color=(mode == "four_color"),
                params=parameter,
                index=index,
            )
            stable_uuid = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"delta:{family}:{candidate_config['name']}:{split_name}:{seed}:{index}",
            )
            problem["id"] = str(stable_uuid)
            problem["difficulty_level"] = candidate_config["name"]
            problem["pattern_type"] = family
            problem["color_mode"] = mode
            wrapped = wrapper.convert_problem(problem)
            wrapped["candidate_id"] = f"{family}__{candidate_config['name']}"
            wrapped["generator_config"] = json.dumps(
                {
                    **candidate_config,
                    "color_mode": mode,
                    "official_parameter": parameter,
                    "generation_seed": item_seed,
                    "split": split_name,
                },
                sort_keys=True,
                default=str,
            )
            rows.append(wrapped)
    return rows


def prepare_target_splits(config: dict[str, Any]) -> dict[str, Any]:
    source = DATA_ROOT / "official_train" / "prepend_sequence_train"
    dataset = load_from_disk(str(source))
    expected = sum(
        int(config["data"][key])
        for key in ("target_train", "initial_audit", "development", "confirmation")
    )
    if len(dataset) != expected:
        raise RuntimeError(f"expected {expected} PREPEND rows, found {len(dataset)}")
    indices = list(range(len(dataset)))
    random.Random(int(config["data"]["split_seed"])).shuffle(indices)
    sizes = [
        ("target_train", int(config["data"]["target_train"])),
        ("initial_audit", int(config["data"]["initial_audit"])),
        ("development", int(config["data"]["development"])),
        ("confirmation", int(config["data"]["confirmation"])),
    ]
    split_root = DATA_ROOT / "target_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": "manufactoria/prepend_sequence_train",
        "source_rows": len(dataset),
        "source_fingerprint": dataset._fingerprint,
        "split_seed": int(config["data"]["split_seed"]),
        "splits": {},
    }
    cursor = 0
    for name, size in sizes:
        selected_indices = indices[cursor : cursor + size]
        cursor += size
        subset = dataset.select(selected_indices)
        path = split_root / name
        subset.save_to_disk(str(path))
        entries = [
            {
                "source_index": source_index,
                "id": subset[i]["id"],
                "name": subset[i]["name"],
                "difficulty": subset[i]["difficulty"],
            }
            for i, source_index in enumerate(selected_indices)
        ]
        write_jsonl(EXP_ROOT / "manifests" / f"{name}.jsonl", entries)
        manifest["splits"][name] = {
            "count": size,
            "data_path": str(path),
            "manifest_path": str(EXP_ROOT / "manifests" / f"{name}.jsonl"),
            "ids": [entry["id"] for entry in entries],
        }
    all_ids = [item for split in manifest["splits"].values() for item in split["ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("target splits overlap")
    atomic_json(EXP_ROOT / "manifests" / "target_split_manifest.json", manifest)
    return manifest


def prepare_candidate_audits(config: dict[str, Any]) -> dict[str, Any]:
    output_root = DATA_ROOT / "candidate_pool"
    output_root.mkdir(parents=True, exist_ok=True)
    families = sorted(
        family for family in GeneratorRegistry.list_pattern_types() if family != "prepend_sequence"
    )
    count = int(config["audit"]["candidate_instances"])
    seed = int(config["data"]["candidate_generation_seed"])
    entries: list[dict[str, Any]] = []
    for family in families:
        for candidate_config in config["candidate_pool"]["configurations"]:
            candidate_id = f"{family}__{candidate_config['name']}"
            rows = generate_distribution(family, candidate_config, count, seed, "audit")
            path = output_root / candidate_id / "audit"
            Dataset.from_list(rows).save_to_disk(str(path))
            write_jsonl(EXP_ROOT / "manifests" / "candidate_audits" / f"{candidate_id}.jsonl", rows)
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "family": family,
                    "configuration": {
                        **candidate_config,
                        "color_mode": effective_color_mode(family, candidate_config["color_mode"]),
                    },
                    "instances": len(rows),
                    "audit_data_path": str(path),
                }
            )
    manifest = {
        "official_generator_root": str(MANU_ROOT / "manufactoria_problem_generators"),
        "official_prompt_wrapper": str(MANU_ROOT / "hf_file_wrapper.py"),
        "generation_seed": seed,
        "families": families,
        "family_count": len(families),
        "entries": entries,
    }
    atomic_json(EXP_ROOT / "manifests" / "candidate_pool_generation_manifest.json", manifest)
    return manifest


def check_prompt_lengths(config: dict[str, Any], candidate_manifest: dict[str, Any]) -> None:
    tokenizer = load_tokenizer("left")
    max_allowed = int(config["sampling"]["max_prompt_tokens"])
    summaries: list[dict[str, Any]] = []
    paths = [
        DATA_ROOT / "target_splits" / name
        for name in ("target_train", "initial_audit", "development", "confirmation")
    ] + [Path(entry["audit_data_path"]) for entry in candidate_manifest["entries"]]
    for path in paths:
        rows = load_from_disk(str(path))
        lengths = [
            len(tokenizer(apply_chat_template(tokenizer, row["messages"])).input_ids)
            for row in rows
        ]
        if max(lengths) > max_allowed:
            raise RuntimeError(f"prompt at {path} exceeds fixed {max_allowed} token limit")
        summaries.append(
            {
                "path": str(path),
                "count": len(lengths),
                "min_tokens": min(lengths),
                "max_tokens": max(lengths),
                "mean_tokens": float(np.mean(lengths)),
            }
        )
    atomic_json(EXP_ROOT / "manifests" / "prompt_length_summary.json", summaries)


def materialize_selected_training(candidate_id: str, common_budget: int = 480) -> Path:
    generation_manifest = read_json(
        EXP_ROOT / "manifests" / "candidate_pool_generation_manifest.json"
    )
    entry = next(
        item for item in generation_manifest["entries"] if item["candidate_id"] == candidate_id
    )
    config = load_config()
    rows = generate_distribution(
        entry["family"],
        entry["configuration"],
        common_budget,
        int(config["data"]["candidate_generation_seed"]),
        "training",
    )
    path = DATA_ROOT / "candidate_pool" / candidate_id / "training"
    Dataset.from_list(rows).save_to_disk(str(path))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize-selected", nargs="*")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    if args.materialize_selected:
        for candidate_id in args.materialize_selected:
            print(materialize_selected_training(candidate_id), flush=True)
        return
    if (EXP_ROOT / "manifests" / "data_prepared.json").exists():
        print("data preparation already complete", flush=True)
        return
    target_manifest = prepare_target_splits(config)
    candidate_manifest = prepare_candidate_audits(config)
    check_prompt_lengths(config, candidate_manifest)
    atomic_json(
        EXP_ROOT / "manifests" / "data_prepared.json",
        {
            "target_split_manifest": str(
                EXP_ROOT / "manifests" / "target_split_manifest.json"
            ),
            "candidate_pool_manifest": str(
                EXP_ROOT / "manifests" / "candidate_pool_generation_manifest.json"
            ),
            "target_source_fingerprint": target_manifest["source_fingerprint"],
        },
    )
    print(
        f"prepared {candidate_manifest['family_count']} non-PREPEND families and target splits",
        flush=True,
    )


if __name__ == "__main__":
    main()
