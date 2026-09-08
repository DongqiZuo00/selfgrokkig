"""Real initial-Solver stage screen; all candidates share the same untouched base."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from stage_catalogue import DEFINITIONS, make_stage_rows, stage_spec
from hf_backend import Solver, INITIAL_SOLVER, sha256, write_json


def main(output, *, grammar=False):
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Do not overwrite a stage screen")
    output.mkdir(parents=True)
    original_hash = sha256(INITIAL_SOLVER / "adapter_model.safetensors")
    tasks = [(item["id"], "concise") for item in DEFINITIONS]
    tasks += [(name, "official") for name in ("identity", "append_R", "append_B")]
    write_json(output / "frozen_screen.json", {"tasks": tasks, "samples_per_task": 8,
             "caps": {"target_small2": 2048, "target_small3": 2048, "default": 768},
             "source_solver": str(INITIAL_SOLVER), "source_adapter_sha256": original_hash,
             "selection_by": "binary stage successes and mixed reward groups; no heldout/test results",
             "reference_programs_used_as_model_input": False,
             "decoding_policy": "dsl_grammar_v1" if grammar else "unconstrained_native",
             "all_screen_samples_use_unchanged_initial_solver": True})
    solver = Solver(grammar_enabled=grammar)
    summaries, first_mixed = [], None
    for index, (name, variant) in enumerate(tasks):
        row = make_stage_rows(name, variant)[0]
        cap = 2048 if name.startswith("target_small") else 768
        path = output / "samples" / f"{index:02d}_{name}_{variant}.json"
        raw = solver.generate(row, 8, cap, 2026090800 + index, output=path)
        result = {"stage_id": name, "variant": variant, "instance_id": row["id"],
                  "parse_valid": raw["parse_valid"], "binary_successes": raw["binary_successes"],
                  "mixed_group": raw["mixed_group"], "generated_tokens": raw["generated_tokens"],
                  "sampling_seconds": raw["sampling_seconds"], "raw_path": str(path)}
        summaries.append(result)
        print(json.dumps(result), flush=True)
        write_json(output / "screen_progress.json", summaries)
        if raw["mixed_group"] and first_mixed is None:
            first_mixed = (result, raw)
    # A real mixed reward update is separately tagged. It cannot contaminate the
    # screen or become the round base by accident; the round uses initial Solver.
    proof = None
    if first_mixed is not None:
        selected, raw = first_mixed
        proof = {"selected_stage": selected, "update": solver.update(raw),
                 "source_adapter": str(INITIAL_SOLVER), "round_base_changed": False}
        proof["checkpoint"] = solver.save(output / "one_binary_update_proof")
        write_json(output / "one_binary_update_proof.json", proof)
        print(json.dumps({"actual_binary_update_proof": proof}), flush=True)
    result = {"status": "completed", "mode": "real_model_initial_stage_screen", "summaries": summaries,
              "rollouts": len(tasks) * 8, "generated_tokens": solver.generated_tokens,
              "mixed_stage_groups": sum(r["mixed_group"] for r in summaries),
              "total_stage_successes": sum(r["binary_successes"] for r in summaries),
              "actual_optimizer_steps": solver.optimizer_steps, "update_proof": proof,
              "source_adapter_unchanged": sha256(INITIAL_SOLVER / "adapter_model.safetensors") == original_hash,
              "target_transfer_or_VERGE_gain_established": False}
    if not result["source_adapter_unchanged"]:
        raise AssertionError("Original Solver adapter changed")
    write_json(output / "SUMMARY.json", result)
    print(json.dumps(result), flush=True)
    solver.close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    main(cli.parse_args().output)
