"""Summarize saved execution evidence; never invent missing run outcomes."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))
from generated_budget import verify_journal


def main():
    cpu = sorted(p for p in (ROOT / "runs").glob("cpu_*/integration/SUMMARY.json"))
    if not cpu:
        raise ValueError("No completed integrated CPU runner evidence")
    integrated = json.loads(cpu[-1].read_text(encoding="utf-8"))
    if integrated["status"] != "PASS" or not integrated["all_telescoping_residuals_zero"]:
        raise ValueError("CPU integration did not pass")
    log_root = cpu[-1].parents[1]
    counts = {}
    for name in ("protocol", "stages", "runtime", "benchmark"):
        text = (log_root / (name + ".log")).read_text(encoding="utf-8")
        matches = re.findall(r"Ran (\d+) tests? in", text)
        if len(matches) != 1 or not text.rstrip().endswith("OK"):
            raise ValueError(f"Incomplete or failed {name} CPU log")
        counts[name] = int(matches[0])
    gpu = {}
    for path in sorted((ROOT / "runs").glob("gpu_smoke*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        events = [json.loads(x) for x in (path.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        verify_journal(path.parent / "events.jsonl")
        sampled = [x["payload"] for x in events if x["kind"] == "sampled_group"]
        generated = sum(len(ids) for group in sampled for ids in group["completion_token_ids"])
        full_pass = sum(sum(group["rewards"]) for group in sampled)
        if summary["status"] != "complete" or generated != summary["generated_tokens"] or full_pass != summary["binary_successes"]:
            raise ValueError(f"GPU summary disagrees with native raw records: {path}")
        gpu[path.parent.name] = {"summary": summary,
            "parse_valid": sum(v["parse_valid"] for group in sampled for v in group["verification"]),
            "all_binary": all(type(r) is int and r in (0, 1) for group in sampled for r in group["rewards"]),
            "journal_verified": True, "summary_matches_native_records": True}
    sources = {}
    for path in sorted(ROOT.rglob("*")):
        if (path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}
                and not any(part in {"generated_24_draft", "__pycache__", "runs"} for part in path.relative_to(ROOT).parts)):
            sources[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    allocation = {}
    for path in (ROOT / "runs").glob("gpu_smoke*_sacct.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("|")
            if not fields[0].isdigit():
                continue
            hours, minutes, seconds = map(int, fields[3].split(":"))
            elapsed = 3600 * hours + 60 * minutes + seconds
            if fields[1:3] != ["COMPLETED", "0:0"] or "gres/gpu:b200=1" not in fields[4]:
                raise ValueError("Smoke Slurm record is not successful single-B200 completion")
            allocation[fields[0]] = {"elapsed_seconds": elapsed, "b200_hours": elapsed / 3600,
                                     "state": fields[1], "exit_code": fields[2], "allocated_tres": fields[4]}
    result = {"status": "saved_evidence_verified", "cpu_tests": counts, "cpu_tests_total": sum(counts.values()),
              "cpu_runner_directory": str(log_root), "synthetic_integration": integrated,
              "gpu_runs": gpu, "slurm_allocations": allocation,
              "total_b200_hours": sum(value["b200_hours"] for value in allocation.values()), "source_sha256": sources,
              "formal_closed_loop_completed": False, "scientific_predictions_established": False}
    (ROOT / "FINAL_VALIDATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cpu_tests": counts,
                      "gpu_runs": {k: {"parse_valid": v["parse_valid"],
                       "successes": v["summary"]["binary_successes"]} for k, v in gpu.items()}}, indent=2))


if __name__ == "__main__":
    main()
