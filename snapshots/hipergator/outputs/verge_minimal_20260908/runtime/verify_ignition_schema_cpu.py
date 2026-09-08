"""Compile/replay schema fixtures on the real CPU grammar worker; no model load."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
import prepare_ignition_round as p
import prepare_named_round as named
os.environ["HF_HOME"] = str(p.WORK / "cache/huggingface")
sys.path.insert(0, str(p.ROOT / "decoding"))
from hf_grammar_bridge import GrammarService


def main(output):
    output = Path(output).resolve()
    p.require(output.is_relative_to(p.WORK.resolve()), "CPU verification outputs must stay inside selfgrok")
    output.mkdir(parents=True, exist_ok=False)
    loaded = named.load_view(named.VIEW)
    screen = named.load_screen(named.SCREEN, loaded["catalogue"])
    eligible, observations = p.observed_starts(loaded["stages"], screen)
    _, fragment, reuse = p.reuse_base(p.PREPARATION, loaded["baseline"])
    p.write_json(output / "actual_baseline_reuse_check.json", reuse)
    schema = p.ignition_schema(loaded["stages"], eligible)
    schema_path = output / "proposal_schema.json"
    p.write_json(schema_path, schema)
    # These are grammar-language fixtures only, never a Q output or plan.
    stages = [{k: s[k] for k in ("stage_id", "kind", "spec")} for s in loaded["stages"]]
    by_id = {s["stage_id"]: s for s in stages}
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(str(p.MODEL), local_files_only=True, fix_mistral_regex=True)
    results = []
    with GrammarService(p.WORK / "envs/vllm/bin/python", p.MODEL, json_schema_path=schema_path, prefix="",
                        log_name="ignition_schema_cpu_verification.stderr.log") as grammar:
        for offset, first in enumerate(eligible):
            rest = [stages[(i + offset) % len(stages)] for i in range(8)]
            fixture = {"base_checkpoint": str(p.INITIAL_SOLVER), "curricula": {
                "g1": [by_id[first], *rest[:2]], "g2": rest[2:5], "g3": rest[5:8]}}
            p.assert_proposal(fixture, loaded["stages"], eligible, str(p.INITIAL_SOLVER))
            path = output / "schema_fixtures" / f"SYNTHETIC_GRAMMAR_ONLY_{first}.json"
            p.write_json(path, fixture)
            ids = tokenizer.encode(json.dumps(fixture, separators=(",", ":")), add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            packed, terminated = grammar.replay(ids)
            indices = np.asarray(ids, dtype=np.int64)
            chosen = (packed[np.arange(len(indices)), indices // 32].astype(np.int64) >> (indices % 32)) & 1
            p.require(bool(chosen.all()) and terminated and packed.shape[0] == len(ids), "legal fixture token replay failed")
            results.append({"fixture": str(path), "native_tokenized_fixture_length": len(ids),
                            "all_tokens_allowed_by_actual_masks": True, "terminated_on_eos": terminated})
            del packed
        illegal = {"base_checkpoint": str(p.INITIAL_SOLVER), "curricula": {
            "g1": [by_id["prepend_RB"], stages[0], stages[1]], "g2": stages[:3], "g3": stages[3:6]}}
        ids = tokenizer.encode(json.dumps(illegal, separators=(",", ":")), add_special_tokens=False) + [tokenizer.eos_token_id]
        rejected = False
        try:
            grammar.replay(ids)
        except ValueError:
            rejected = True
        p.require(rejected, "grammar accepted a first stage outside the declared prior")
        metadata = grammar.metadata
    summary = {"status": "pass", "mode": "cpu_schema_fixture_and_readonly_real_baseline_check",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "model_weights_loaded": False,
        "new_model_samples": 0, "new_training_updates": 0, "schema_source": named.source(schema_path),
        "entry_source": named.source(Path(p.__file__)), "grammar_metadata": metadata,
        "legal_fixture_replays": results, "disallowed_first_stage_rejected": rejected,
        "observed_start_candidates": observations, "actual_base_reused_records": len(fragment["evaluations"]["base"]["records"]),
        "new_base_rollouts": 0, "new_base_generated_tokens": 0,
        "fixture_is_challenger_output": False, "fixture_is_experiment_result": False}
    p.write_json(output / "SUMMARY.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    main(cli.parse_args().output)
