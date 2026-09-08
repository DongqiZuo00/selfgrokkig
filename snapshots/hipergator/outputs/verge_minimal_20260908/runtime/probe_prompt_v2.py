"""Paired stage screen: change only the generic grammar naming instructions."""
import argparse
import json
from pathlib import Path
import re
from hf_backend import ROOT, Solver, INITIAL_SOLVER, sha256, write_json

TASKS = ROOT / "benchmarks/generated/naming_prompt_v2/screen_tasks.json"


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(TASKS.read_text(encoding="utf-8"))
    original_hash = sha256(INITIAL_SOLVER / "adapter_model.safetensors")
    write_json(output / "frozen_screen.json", {"task_file": str(TASKS), "task_file_sha256": sha256(TASKS),
        "screen": frozen, "original_adapter_sha256": original_hash,
        "same_initial_solver_same_seeds_same_caps_same_tests_same_grammar_as_v1": True,
        "changed": "only generic canonical node-name instructions", "reference_programs_in_prompts": False})
    solver = Solver(grammar_enabled=True)
    summaries, first_mixed = [], None
    try:
        for index, task in enumerate(frozen["tasks"]):
            path = output / "samples" / f"{index:02d}_{task['stage_id']}_{task['variant']}.json"
            raw = solver.generate(task["row"], 8, task["cap"], task["seed"], output=path)
            routes = []
            for text in raw["verifier_completions"]:
                match = re.search(r"START start:\s*\n\s*NEXT\s+(\S+)", text)
                routes.append(match.group(1) if match else None)
            summary = {"stage_id": task["stage_id"], "variant": task["variant"],
                "parse_valid": raw["parse_valid"], "binary_successes": raw["binary_successes"],
                "mixed_group": raw["mixed_group"], "generated_tokens": raw["generated_tokens"],
                "first_routes": routes, "raw_path": str(path), "sampling_seconds": raw["sampling_seconds"]}
            summaries.append(summary)
            write_json(output / "screen_progress.json", summaries)
            print(json.dumps(summary), flush=True)
            if raw["mixed_group"] and first_mixed is None:
                first_mixed = summary, raw
        proof = None
        if first_mixed:
            selected, raw = first_mixed
            proof = {"selected_stage": selected, "update": solver.update(raw), "round_base_changed": False,
                     "checkpoint": solver.save(output / "one_binary_update_proof")}
            write_json(output / "one_binary_update_proof.json", proof)
        result = {"status": "completed", "mode": "real_model_naming_only_paired_stage_screen",
            "summaries": summaries, "rollouts": len(summaries) * 8,
            "generated_tokens": solver.generated_tokens, "parse_valid": sum(r["parse_valid"] for r in summaries),
            "binary_successes": sum(r["binary_successes"] for r in summaries),
            "mixed_groups": sum(r["mixed_group"] for r in summaries), "actual_optimizer_steps": solver.optimizer_steps,
            "update_proof": proof, "source_adapter_unchanged": sha256(INITIAL_SOLVER / "adapter_model.safetensors") == original_hash,
            "target_transfer_or_VERGE_gain_established": False}
        if not result["source_adapter_unchanged"]:
            raise AssertionError("Original adapter changed")
        write_json(output / "SUMMARY.json", result)
        print(json.dumps(result), flush=True)
    finally:
        solver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
