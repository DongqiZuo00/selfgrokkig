"""Audit actual generation and scheduler cost for explicitly named selfgrok runs."""
import argparse
import json
from pathlib import Path
import subprocess
from hf_backend import ROOT, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def native_tokens(path):
    record = read(path)
    values = record["completion_token_ids"]
    result = sum(map(len, values)) if values and isinstance(values[0], list) else len(values)
    if record.get("generated_tokens") != result:
        raise ValueError(f"Native token count differs from source report: {path}")
    return result


def main(output):
    runs = ROOT / "runs"
    costs = {}
    for label, relative, pattern in (
        ("unconstrained_stage_screen", "stage_screen_retry1_41410864/samples", "*.json"),
        ("grammar_stage_screen", "stage_grammar_v1_41412533/samples", "*.json"),
        ("naming_v2_stage_screen", "naming_v2_41415498/samples", "*.json"),
        ("round1_baseline", "round_prepare_41413994/base/raw_evaluation", "*.json"),
        ("round1_challenger_all_attempts", "round_prepare_41413994", "challenger_sample_*.json"),
    ):
        files = sorted((runs / relative).glob(pattern))
        costs[label] = {"raw_files": len(files), "actual_generated_tokens": sum(native_tokens(path) for path in files)}
    for index, branch in enumerate(("g1", "g2", "g3", "direct")):
        directory = runs / f"round1_{branch}_41414619_{index}"
        training = list((directory / "execution/raw_training").glob("*.json"))
        evaluation = list((directory / "execution/raw_evaluation").glob("*.json"))
        count = sum(native_tokens(path) for path in training)
        completed = (directory / "COMPLETE.json").exists()
        if completed:
            fragment = read(directory / "fragment.json")
            if fragment["execution"]["training_generated_tokens"] != count or count != 65536:
                raise ValueError("Completed branch actual native count differs from phase ledger/B")
        costs[f"round1_{branch}"] = {"complete": completed, "training_raw_files": len(training),
                                    "training_actual_generated_tokens": count,
                                    "new_evaluation_raw_files": len(evaluation),
                                    "new_evaluation_actual_generated_tokens": sum(native_tokens(p) for p in evaluation)}
    jobs = "41410512,41410864,41412533,41413994,41414619,41415498"
    command = ["sacct", "-j", jobs, "--parsable2", "--noheader", "--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode"]
    completed = subprocess.run(command, text=True, check=True, capture_output=True)
    scheduler = []
    for line in completed.stdout.splitlines():
        job, state, seconds, tres, exit_code = line.split("|")
        if "." in job or "[" in job:
            continue
        resources = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
        gpus = int(resources.get("gres/gpu", 0))
        scheduler.append({"job_id": job, "state": state, "elapsed_seconds": int(seconds),
                          "allocated_tres": resources, "b200_hours_elapsed": int(seconds) * gpus / 3600,
                          "exit_code": exit_code})
    total = sum(value.get("actual_generated_tokens", 0) + value.get("training_actual_generated_tokens", 0) +
                value.get("new_evaluation_actual_generated_tokens", 0) for value in costs.values())
    result = {"mode": "actual_saved_native_tokens_and_scheduler_allocation", "generation": costs,
              "actual_generated_tokens_saved": total, "scheduler": scheduler,
              "b200_hours_elapsed": sum(row["b200_hours_elapsed"] for row in scheduler),
              "all_named_jobs_terminal": all(row["state"] not in ("RUNNING", "PENDING", "COMPLETING") for row in scheduler),
              "scheduler_command": command, "cached_evaluations_add_zero_new_generation": True,
              "stage_update_proof_reuses_screen_tokens_no_new_generation": True,
              "excludes_previous_release_smokes_and_other_projects": True}
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
