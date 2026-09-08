"""Local delivery acceptance from complete, already independently audited real evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


if __name__ == "__main__":
    results = [read(ROOT / f"runs/round{i}_result/summary.json") for i in (1, 2, 3)]
    for index, result in enumerate(results, 1):
        require(result["status"] == "complete" and result["mode"] == "real_model_round", "real complete rounds required")
        require(result["total_training_generated_tokens"] == 4 * 65536, "round training budget differs")
        require(read(ROOT / f"evidence/round{index}_independent_score_audit.json")["status"] == "PASS", "independent score audit missing")
    final = results[-1]
    require(final["at_least_one_curriculum_mixed_update"] and final["real_curriculum_mixed_binary_optimizer_steps"] == 1,
            "stage probe alone cannot satisfy real curriculum update requirement")
    checkpoint = ROOT / "runs/round3_g1_41420288_0/execution/checkpoints/g1_m1/adapter_model.safetensors"
    expected = "b5278b1fed5e784945b6770b9abfd3b8d0cede946d2d98ffd771871f2fc86090"
    require(sha(checkpoint) == expected, "local trained checkpoint differs from independently compared remote tensor file")
    exchange = read(ROOT / "runs/round3_result/exchange.json")
    unique = {}
    for item in exchange["evaluations"].values():
        for record in item.get("records", []):
            key = record["generation_evidence_id"]
            if key in unique:
                require(record == unique[key], "shared raw source has inconsistent records")
            unique[key] = record
    require(len(unique) == 512, "complete reused base and new trained evaluation expected")
    target_eval_successes = sum(r["conditions"][-1] for r in unique.values())
    require(target_eval_successes == 0, "review target first-success observations before reporting")
    events = read(ROOT / "runs/round3_result/events.json")
    require(all(not v for v in events["training_target_successes_by_branch"].values()), "training target success needs reporting")
    reuse = read(ROOT / "runs/round_prepare_ignition_41419999/baseline_reuse.json")
    require(reuse["new_evaluation_rollouts"] == reuse["new_evaluation_generated_tokens"] == 0, "baseline reuse cost differs")
    require(final["evaluation_generated_tokens"] - reuse["original_evaluation_generated_tokens"] == 61055,
            "new versus referenced evaluation cost differs")
    costs = read(ROOT / "evidence/COST_THREE_ROUNDS.json")
    require(costs["status"] == "complete", "cost and terminal scheduler audit must finish")
    report = {"status": "PASS", "minimal_real_process_experiment_complete": True,
              "target_solved": False, "method_effectiveness_established": False,
              "training_generated_tokens_per_branch": 65536, "third_round_curriculum_updates": 1,
              "third_round_distinct_target_evaluation_samples_referenced": len(unique),
              "third_round_new_target_evaluation_samples": 256, "third_round_new_target_evaluation_tokens": 61055,
              "third_round_target_successes_all_evaluation_records": target_eval_successes,
              "target_training_success_event_observed": False,
              "global_cross_job_first_success_order_claimed": False,
              "third_round_gains": read(ROOT / "runs/round3_result/round_score.json"),
              "third_round_archive_cells": final["archive_cells"], "third_round_lead": final["lead_checkpoint"],
              "local_changed_checkpoint": str(checkpoint), "local_changed_checkpoint_sha256": expected,
              "formal_predictions_verified": [], "persistent_multi_round_archive_rft_production_complete": False,
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                  [ROOT / f"runs/round{i}_result/summary.json" for i in (1, 2, 3)] +
                  [ROOT / "evidence/COST_THREE_ROUNDS.json", ROOT / "evidence/round3_independent_score_audit.json"]}}
    path = ROOT / "EXPERIMENT_RESULT.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "result": str(path), "actual_curriculum_updates": 1,
                      "target_successes": 0, "gamma": [0, 0, 0], "effective_method_claim": False}))
