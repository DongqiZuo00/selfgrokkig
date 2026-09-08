"""Post-hoc diagnostics of three physical selection evaluation datasets only.

Does not change rewards, classes, frozen inputs, training or model outputs.
Optional CPU interpretation uses the original vendor on the same saved draws.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import stage_catalogue as stages

SETS = (
    ("round1_initial_compact", "runs/round_prepare_41413994/base", "base", "benchmarks/generated/pilot_view_v1"),
    ("round2_initial_named", "runs/round_prepare_named_41417194/base", "base", "benchmarks/generated/naming_round_v2"),
    ("round3_g1_updated_named", "runs/round3_g1_41420288_0/execution", "g1_final", "benchmarks/generated/naming_round_v2"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_cycle(factory):
    edges = {k: {r.target for r in n.routes if r.target in factory.nodes} for k, n in factory.nodes.items()}
    indegree = dict.fromkeys(edges, 0)
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    stack = [k for k, degree in indegree.items() if degree == 0]
    visited = 0
    while stack:
        source = stack.pop()
        visited += 1
        for target in edges[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                stack.append(target)
    return visited != len(edges)


def analyze(parser=None):
    vendor = stages.load_module(parser, "target_eval_posthoc_vendor") if parser else None
    result, pairing = [], {}
    source_hashes = {}
    for name, directory, alias, view in SETS:
        run, view_dir = ROOT / directory, ROOT / view
        rows = {r["id"]: r for r in read(view_dir / "target_rows.json")["selection"]}
        meta = read(view_dir / "target_view_metadata.json")
        eval_path = run / "evaluations" / f"{alias}.json"
        evaluated = read(eval_path)
        evaluations = {(r["instance_id"], r["chunk_index"], r["sample_index_in_chunk"]): r for r in evaluated["records"]}
        assert len(evaluations) == 256 and len(rows) == 8
        counts, failures, f_distribution, cycle_counts = Counter(), Counter(), Counter(), Counter()
        class_pass, class_total, class_any, class_all = Counter(), Counter(), Counter(), Counter()
        semantic, semantic_denominators = Counter(), Counter()
        semantic_examples, raw_files, draws = [], [], []
        pair_contract = {}
        paths = sorted((run / "raw_evaluation").glob("*.json"))
        assert len(paths) == 32
        for path in paths:
            raw = read(path)
            row = rows[raw["instance_id"]]
            assert row.get("hint") is None and all(stages.TARGET.split_of(c["input"]) == "selection" for c in row["ground_truth"])
            assert len(row["ground_truth"]) == 36 and Counter(c["test_class"] for c in row["ground_truth"]) == Counter(dict.fromkeys(stages.TARGET.CLASS_IDS, 6))
            assert row["messages"][0]["content"] in raw["prompt"] and raw["n"] == 8
            chunk = int(re.search(r"__chunk(\d+)\.json$", path.name).group(1))
            raw_hash = digest(path)
            source_hashes[str(path.relative_to(ROOT))] = raw_hash
            raw_files.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(ROOT)), "sha256": raw_hash,
                "instance_id": row["id"], "chunk": chunk, "seed": raw["seed"], "n": raw["n"]})
            pair_contract[(row["id"], chunk)] = (raw["seed"], raw["max_new_tokens"], raw["prompt"], raw["grammar_metadata"]["grammar_sha256"], row["ground_truth"])
            counts["generated_tokens"] += sum(map(len, raw["completion_token_ids"]))
            assert sum(map(len, raw["completion_token_ids"])) == raw["generated_tokens"]
            for index, (saved, text, ids) in enumerate(zip(raw["verification"], raw["verifier_completions"], raw["completion_token_ids"])):
                record = evaluations[(row["id"], chunk, index)]
                assert record["raw_generation_sha256"] == raw_hash
                assert record["raw_completion"] == text and record["completion_token_ids"] == ids and record["hint"] is None
                assert record["prompt_view_sha256"] == meta["prompt_view_sha256"]
                assert record["chunk_seed"] == raw["seed"]
                assert len(saved["per_test"]) == 36
                pass_by_class = Counter()
                for case, outcome in zip(row["ground_truth"], saved["per_test"]):
                    assert case["input"] == outcome["input"] and case["test_class"] == outcome["test_class"]
                    assert record["test_outcomes"][case["test_id"]] == bool(outcome["pass"])
                    k = case["test_class"]
                    class_total[k] += 1
                    class_pass[k] += outcome["pass"]
                    pass_by_class[k] += outcome["pass"]
                    if not outcome["pass"]:
                        failures[outcome["reason"]] += 1
                rates = {k: pass_by_class[k] / 6 for k in stages.TARGET.CLASS_IDS}
                f = min(rates.values())
                assert saved["f"] == f and saved["class_rates"] == rates
                assert saved["reward"] == int(all(c["pass"] for c in saved["per_test"]))
                counts.update(programs=1, test_case_executions=36, parse_valid=saved["parse_valid"],
                    binary_full_pass=saved["reward"], case_exact_pass=sum(pass_by_class.values()),
                    any_case_pass=int(any(pass_by_class.values())))
                f_distribution[str(f)] += 1
                cycle_counts[f"saved_static_cycle_{record['has_cycle']}"] += 1
                for k in stages.TARGET.CLASS_IDS:
                    class_any[k] += int(pass_by_class[k] > 0)
                    class_all[k] += int(pass_by_class[k] == 6)
                draw = {"instance_id": row["id"], "chunk": chunk, "sample_index": index,
                    "f": f, "class_pass_counts": dict(pass_by_class), "static_cycle": record["has_cycle"]}
                if vendor:
                    factory = vendor.create_robot_factory(stages.TARGET.extract_program(text))
                    static = static_cycle(factory)
                    assert static == record["has_cycle"]
                    dynamic_any = dynamic_finished = False
                    for ci, (case, expected_record) in enumerate(zip(row["ground_truth"], saved["per_test"])):
                        actual = factory.process_robot(case["input"])
                        passed = bool(actual.finished and actual.final_tape == case["expected_output"])
                        reason = actual.rejection_reason if not actual.finished else (None if passed else "output_mismatch")
                        assert int(passed) == expected_record["pass"] and len(actual.path) == expected_record["steps"] and reason == expected_record["reason"]
                        repeated = len(actual.path) != len(set(actual.path))
                        dynamic_any |= repeated
                        dynamic_finished |= repeated and actual.finished
                        cycle_counts["case_with_node_revisit"] += int(repeated)
                        cycle_counts["finished_case_with_node_revisit"] += int(repeated and actual.finished)
                        semantic_denominators["all_cases"] += 1
                        mutation_needed = "RB" in case["input"]
                        semantic_denominators["mutation_needed_cases"] += int(mutation_needed)
                        if not actual.finished:
                            continue
                        semantic_denominators["finished_cases"] += 1
                        semantic_denominators["finished_mutation_needed_cases"] += int(mutation_needed)
                        out, replacement = actual.final_tape, case["input"].replace("RB", "YG")
                        prefix_ok = out.startswith("RBYGRB")
                        replacement_only = out == replacement
                        payload_after_six = len(out) >= 6 and out[6:] == replacement
                        semantic["finished_output_has_required_prefix"] += int(prefix_ok)
                        semantic["finished_output_equals_replacement_only"] += int(replacement_only)
                        semantic["replacement_only_on_mutation_needed_input"] += int(replacement_only and mutation_needed)
                        semantic["finished_output_tail_after_six_equals_replacement"] += int(payload_after_six)
                        semantic["tail_after_six_correct_on_mutation_needed_input"] += int(payload_after_six and mutation_needed)
                        semantic["finished_output_equals_original_input"] += int(out == case["input"])
                        if (prefix_ok or (replacement_only and mutation_needed) or (payload_after_six and mutation_needed)) and len(semantic_examples) < 20:
                            semantic_examples.append({"source": str(path.relative_to(ROOT)), "sample_index": index,
                                "case_index": ci, "input": case["input"], "actual_finished_output": out,
                                "target_expected": case["expected_output"], "prefix_correct": prefix_ok,
                                "replacement_only": replacement_only, "payload_after_six_correct": payload_after_six})
                    cycle_counts["program_has_node_revisit_on_any_case"] += int(dynamic_any)
                    cycle_counts["program_has_finished_node_revisit_on_any_case"] += int(dynamic_finished)
                    cycle_counts["static_cycle_but_no_selected_case_revisits_node"] += int(static and not dynamic_any)
                    draw.update(dynamic_node_revisit=dynamic_any, finished_dynamic_node_revisit=dynamic_finished)
                draws.append(draw)
        assert counts["programs"] == 256 and counts["test_case_executions"] == 9216
        source_hashes[str(eval_path.relative_to(ROOT))] = digest(eval_path)
        source_hashes[str((view_dir / "target_rows.json").relative_to(ROOT))] = digest(view_dir / "target_rows.json")
        result.append({"dataset": name, "counts": dict(counts), "f_distribution": dict(f_distribution),
            "class_diagnostics": {k: {"exact_pass": class_pass[k], "cases": class_total[k], "programs_with_any_pass": class_any[k],
                "programs_with_all_six_pass": class_all[k]} for k in stages.TARGET.CLASS_IDS},
            "failure_reasons": dict(failures), "cycle_diagnostics": dict(cycle_counts),
            "posthoc_output_diagnostics": dict(semantic), "posthoc_denominators": dict(semantic_denominators),
            "posthoc_examples": semantic_examples, "checkpoint": evaluated["checkpoint"], "weights_sha256": evaluated["weights_sha256"],
            "prompt_view_sha256": meta["prompt_view_sha256"], "manifest_id": meta["selection"]["manifest_id"],
            "physical_raw_files": raw_files, "draw_diagnostics": draws})
        pairing[name] = pair_contract
    assert pairing["round2_initial_named"] == pairing["round3_g1_updated_named"]
    assert result[1]["prompt_view_sha256"] == result[2]["prompt_view_sha256"] != result[0]["prompt_view_sha256"]
    assert result[1]["weights_sha256"] != result[2]["weights_sha256"] and result[0]["weights_sha256"] == result[1]["weights_sha256"]
    return {"status": "PASS", "mode": "existing_physical_selection_eval_posthoc_diagnostics", "datasets": result,
        "physical_datasets": 3, "physical_raw_groups": 96, "physical_model_draws": 768, "case_executions": 27648,
        "cache_deduplication": "Count exactly 32 raw generation groups per physical checkpoint/view dataset. g1_m1 chunk0 plus g1_final chunks1-3 form one updated P32 dataset; do not recount g1_m2/m3 or other alias caches. Equal completion text from distinct sample slots remains distinct draws.",
        "r2_vs_r3_same_prompt_seed_cap_grammar_and_ground_truth": True, "r2_vs_r3_checkpoint_changed": True,
        "r1_vs_r2_prompt_view_changed_same_initial_weights": True,
        "selection_only_verified": True, "heldout_test_model_results_read": False,
        "new_rewards_or_test_classes": False, "new_model_generations": 0, "new_GPU_jobs": 0,
        "cycle_diagnostic_definition": "Static means a directed cycle anywhere in parsed graph, including unreachable nodes; dynamic means a node ID is revisited on an actual vendor path. Node revisits do not by themselves prove identical machine state or useful algorithmic loops.",
        "output_diagnostics_definition": "Only finished executions count. Required prefix is RBYGRB; replacement-only is exact x.replace(RB,YG) without prefix; payload-after-six strips six output symbols without requiring the prefix to be correct. Mutation-needed subsets use existing RB occurrence semantics, not new formal classes or rewards. These are post-hoc behavior checks, not generalization proofs.",
        "parser_sha256": digest(parser) if parser else None, "source_sha256": source_hashes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.parser)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "physical_model_draws": result["physical_model_draws"],
        "case_executions": result["case_executions"], "datasets": [{k: d[k] for k in ("dataset", "counts", "f_distribution", "failure_reasons", "cycle_diagnostics", "posthoc_output_diagnostics", "posthoc_denominators")} for d in result["datasets"]]}, indent=2))
