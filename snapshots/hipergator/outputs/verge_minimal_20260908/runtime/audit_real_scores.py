"""Independent stdlib audit from actual raw per-test outcomes, without scorer imports."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(preparation, result):
    plan = read(preparation / "plan.json")
    exchange = read(result / "exchange.json")
    score = read(result / "round_score.json")
    delta = read(result / "delta.json")
    hashes, sources, resolved = {}, {}, {}

    def digest(path):
        path = Path(path)
        if str(path) not in hashes:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[str(path)]

    def records(alias, seen=()):
        check(alias not in seen, "cyclic evaluation cache")
        value = exchange["evaluations"][alias]
        if "records" in value:
            return value["records"]
        parent = exchange["evaluations"][value["source_evidence_id"]]
        check(parent["weights_sha256"] == value["weights_sha256"] and parent["checkpoint"] == value["checkpoint"],
              "cache crosses checkpoint identity")
        return [r for r in records(value["source_evidence_id"], (*seen, alias)) if r["sample_id"] in value["sample_ids"]]

    for alias, evaluation in exchange["evaluations"].items():
        check(digest(Path(evaluation["checkpoint"]) / "adapter_model.safetensors") == evaluation["weights_sha256"],
              "actual checkpoint weights changed")
        actual = records(alias)
        expected = {(i, s) for i in plan["selection"]["instance_ids"] for s in evaluation["sample_ids"]}
        check(len(actual) == len(expected) and {(r["instance_id"], r["sample_id"]) for r in actual} == expected,
              "missing/duplicate actual selection slot")
        for record in actual:
            check(record["hint"] is None and record["synthetic_fixture"] is False, "real unhinted target required")
            check(record["prompt_view_sha256"] == plan["selection"]["prompt_view_sha256"], "wrong prompt view")
            source_path, slot = record["generation_evidence_id"].rsplit("#sample", 1)
            slot = int(slot)
            check(digest(source_path) == record["raw_generation_sha256"], "raw generation changed")
            if source_path not in sources:
                sources[source_path] = read(source_path)
            raw = sources[source_path]
            check(raw["mode"] == "real_model_generation" and raw["decoding_policy"] == "dsl_grammar_v1", "wrong raw source")
            check(raw["completion_token_ids"][slot] == record["completion_token_ids"], "native token IDs differ")
            check(raw["verifier_completions"][slot] == record["raw_completion"], "verifier text differs")
            check(raw["seed"] == record["chunk_seed"], "actual sample seed differs")
            contract = plan["selection"]["test_contracts"][record["instance_id"]]
            denominators = Counter(contract["class_by_test"].values())
            check(len(denominators) == 6 and set(denominators.values()) == {6}, "changed or empty test class")
            check(set(record["test_outcomes"]) == set(contract["test_ids"]) and len(contract["test_ids"]) == 36,
                  "target test omitted")
            verification = raw["verification"][slot]
            check(len(verification["per_test"]) == 36, "raw verifier omitted test")
            passed = Counter()
            for test_id, outcome in zip(contract["test_ids"], verification["per_test"]):
                check(type(outcome["pass"]) is int and outcome["pass"] in (0, 1), "nonbinary outcome")
                check(record["test_outcomes"][test_id] == bool(outcome["pass"]), "stored per-test result differs")
                category = contract["class_by_test"][test_id]
                check(outcome["test_class"] == category, "raw class differs from frozen mapping")
                passed[category] += outcome["pass"]
            f = min(Fraction(passed[k], n) for k, n in denominators.items())
            calculated = [bool(verification["parse_valid"])] + [f >= t for t in
                (Fraction(1, 36), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))]
            check(calculated == record["conditions"], "condition vector differs from raw min-per-class")
            check(raw["rewards"][slot] == int(calculated[-1]), "Solver target reward differs from full pass")
        resolved[alias] = actual

    def rate(alias, rung, first8=False):
        rows = resolved[alias]
        if first8:
            rows = [r for r in rows if r["sample_id"] in plan["selection"]["sample_ids"][:8]]
        check(rows, "empty evaluation cannot be zero-filled")
        return Fraction(sum(r["conditions"][rung - 1] for r in rows), len(rows))

    branches = ("g1", "g2", "g3", "direct")
    successes = {b: sum(r["conditions"][-1] for r in resolved[f"{b}_final"]) for b in branches}
    base_frontier = next((j for j in range(1, 7) if rate("base", j) < Fraction(9, 10)), 7)
    kappa = 6 if sum(s >= plan["q"] for s in successes.values()) >= 2 else base_frontier
    check(kappa == score["kappa"], "reward switch or frontier differs")
    gains = {}
    for branch in branches[:3]:
        gamma = rate(f"{branch}_final", kappa) - rate("direct_final", kappa)
        gaps = [Fraction(0)] + [rate(f"{branch}_m{j}", kappa) - rate(f"direct_m{j}", kappa) for j in (1, 2, 3)]
        gaps.append(rate(f"{branch}_final", kappa, True) - rate("direct_final", kappa, True))
        stages = [float(b - a) for a, b in zip(gaps, gaps[1:])]
        correction = gamma - gaps[-1]
        check(abs(float(gamma) - score["gains"][branch]["estimate"]) < 1e-12, "Gamma differs")
        check(all(abs(a - b) < 1e-12 for a, b in zip(stages, delta[branch]["deltas"])) and
              len(delta[branch]["deltas"]) == 4, "stage/tail Delta differs")
        check(abs(float(correction) - delta[branch]["sampling_correction"]) < 1e-12, "P32-P8 correction differs")
        check(abs(sum(stages) + float(correction) - float(gamma)) < 1e-12, "telescoping endpoint differs")
        gains[branch] = {"gamma": float(gamma), "stage_deltas_including_tail": stages,
                         "sampling_correction": float(correction)}
    return {"status": "PASS", "round_id": plan["round_id"], "checks":
            "raw SHA/native tokens/full 36 tests/six nonempty classes/min-f conditions/checkpoint weights/cache identity/Gamma/tail Delta/P32-P8",
            "logical_slots_checked": sum(map(len, resolved.values())), "distinct_raw_batches_checked": len(sources),
            "base_rates": [float(rate("base", j)) for j in range(1, 7)],
            "terminal_rates": {b: [float(rate(f"{b}_final", j)) for j in range(1, 7)] for b in branches},
            "terminal_success_counts": successes, "kappa": kappa, "gains": gains,
            "source_sha256": hashes, "new_gpu_work": False,
            "limitations": "Recomputes aggregation from recorded vendor outcomes, not a fresh parser replay or independent sampling; bootstrap CI separately covered by protocol tests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.preparation, args.result)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))
