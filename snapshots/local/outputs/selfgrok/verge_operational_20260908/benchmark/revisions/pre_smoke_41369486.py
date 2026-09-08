"""New, versioned RB-input adaptation of VERGE's existing fixed target.

Only CPU benchmark preparation/verification. No model results, optimizer,
training launch, teacher traces, or hinted evaluation are loaded by this file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

VERSION = "verge_prepend_rb_v1_20260908"
PREFIX = "RBYGRB"
INPUT_ALPHABET = "RB"
DSL_ALPHABET = "RBYG"
MAX_LENGTH = 64
PER_CLASS = 6
CLASS_IDS = tuple(f"matches_{n}__pending_r_{r}" for n in ("0", "1", "2plus") for r in (0, 1))
SPLIT_COUNTS = {"train": 128, "selection": 64, "heldout": 64, "test": 64}
PARAMETERS = {"prepend_sequence": PREFIX, "mutation_type": "replace_pattern_to_pattern",
              "source_pattern": "RB", "target_pattern": "YG"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_utf8(path, text):
    # Byte-stable manifests/data on Windows and Linux (no newline translation).
    path.write_bytes(text.encode("utf-8"))


def expected_output(tape):
    validate_input(tape)
    return PREFIX + tape.replace("RB", "YG")


def validate_input(tape):
    if not isinstance(tape, str) or set(tape) - set(INPUT_ALPHABET) or len(tape) > MAX_LENGTH:
        raise ValueError("input must belong to {R,B}^{0..64}")


def test_class(tape):
    """Disjoint semantic strata fixed before any model sampling.

    Python str.count counts the same nonoverlapping RB occurrences as replace.
    RB has no proper self-overlap. A final R leaves the transducer pending at EOF.
    Empty belongs to matches_0__pending_r_0.
    """
    validate_input(tape)
    count = tape.count("RB")
    category = str(count) if count < 2 else "2plus"
    return f"matches_{category}__pending_r_{int(tape.endswith('R'))}"


def split_of(tape):
    """A tape's split is a function of input alone; no outputs or model scores."""
    validate_input(tape)
    bucket = int(hashlib.sha256((VERSION + "|input|" + tape).encode()).hexdigest(), 16) % 100
    return "train" if bucket < 50 else "selection" if bucket < 70 else "heldout" if bucket < 85 else "test"


def input_support():
    # Complete short strings expose every automaton transition. All B*R* words
    # prevent the rare no-match strata from disappearing under random sampling.
    values = {"".join(bits) for n in range(13) for bits in itertools.product(INPUT_ALPHABET, repeat=n)}
    values.update("B" * a + "R" * b for a in range(MAX_LENGTH + 1) for b in range(MAX_LENGTH + 1 - a))
    rng = random.Random(2026090801)
    for _ in range(4096):
        values.add("".join(rng.choice(INPUT_ALPHABET) for _ in range(rng.randint(13, MAX_LENGTH))))
    values.update("RB" * n + suffix for n in range(1, 32) for suffix in ("", "R", "B"))
    return sorted(values, key=lambda tape: (len(tape), tape))


def load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_program(prefix=PREFIX):
    # Executable proof witness only: this DSL is never inserted in target prompts
    # or supplied as Solver supervised data by this benchmark module.
    colors = {"R": "RED", "B": "BLUE", "Y": "YELLOW", "G": "GREEN"}
    lines = ["START start:", "    NEXT delimiter", "", "PAINTER_YELLOW delimiter:", "    NEXT prefix_0", ""]
    for i, color in enumerate(prefix):
        lines += [f"PAINTER_{colors[color]} prefix_{i}:", f"    NEXT {'scan' if i == len(prefix)-1 else 'prefix_'+str(i+1)}", ""]
    lines += [
        "PULLER_RB scan:", "    [R] pending_r", "    [B] emit_b", "    [EMPTY] finish", "",
        "PAINTER_BLUE emit_b:", "    NEXT scan", "",
        "PULLER_RB pending_r:", "    [R] emit_r_pending", "    [B] emit_y", "    [EMPTY] finish_pending", "",
        "PAINTER_RED emit_r_pending:", "    NEXT pending_r", "",
        "PAINTER_YELLOW emit_y:", "    NEXT emit_g", "",
        "PAINTER_GREEN emit_g:", "    NEXT scan", "",
        "PULLER_YG finish:", "    [Y] end", "    [G] NONE", "    [EMPTY] NONE", "",
        "PULLER_YG finish_pending:", "    [Y] emit_last_r", "    [G] NONE", "    [EMPTY] NONE", "",
        "PAINTER_RED emit_last_r:", "    NEXT end", "", "END end", "",
    ]
    return "\n".join(lines)


def evaluate_program(program, cases, create_factory):
    """Target verifier adapter: stable per-class diagnostics + binary Solver reward.

    create_factory must be the pinned vendor create_robot_factory. Class metadata
    must survive dataset loading; the legacy wrapper drops it, so it cannot be
    round-tripped through _extract_test_cases after benchmark generation.
    """
    if not cases or Counter(c.get("test_class") for c in cases) != Counter(dict.fromkeys(CLASS_IDS, PER_CLASS)):
        raise ValueError(f"target suite must have exactly {PER_CLASS} tests in each frozen class")
    if len({c["input"] for c in cases}) != len(cases):
        raise ValueError("duplicate input in target suite")
    for case in cases:
        if case.get("expected_accepted") is not True or case.get("check_output") is not True:
            raise ValueError("this target requires exact output and successful termination")
        if case.get("test_class") != test_class(case["input"]) or case.get("expected_output") != expected_output(case["input"]):
            raise ValueError("test metadata disagrees with frozen target")
    try:
        factory = create_factory(program)
        parse_valid, parse_error = True, None
    except Exception as error:
        factory, parse_valid, parse_error = None, False, str(error)
    outcomes = []
    for case in cases:
        value, reason, steps = 0, parse_error, 0
        if factory is not None:
            try:
                result = factory.process_robot(case["input"])
                value = int(result.finished and result.final_tape == case["expected_output"])
                reason = result.rejection_reason if not result.finished else (None if value else "output_mismatch")
                steps = len(result.path)
            except Exception as error:
                reason = str(error)
        outcomes.append({"input": case["input"], "test_class": case["test_class"],
                         "pass": value, "reason": reason, "steps": steps})
    rates = {cls: sum(r["pass"] for r in outcomes if r["test_class"] == cls) / PER_CLASS for cls in CLASS_IDS}
    return {"parse_valid": parse_valid, "per_test": outcomes, "class_rates": rates,
            "f": min(rates.values()), "reward": int(all(r["pass"] for r in outcomes))}


def prepare(out, wrapper_path):
    out.mkdir(parents=True, exist_ok=True)
    wrapper = load_python(wrapper_path, "verge_benchmark_prompt_wrapper").TrainingFileWrapper()
    prompt = wrapper._get_prompt_template(True).replace("{criteria}", (
        "The input tape contains only R and B (including the empty tape). "
        "Replace every nonoverlapping occurrence of RB with YG, preserving the order of other symbols; "
        "then prepend RBYGRB. All valid inputs must reach end with exactly this output. "
        "All four standard tape symbols R, B, Y, G and all standard DSL node types remain available."
    ))
    support = input_support()
    pools = defaultdict(list)
    for tape in support:
        pools[split_of(tape), test_class(tape)].append(tape)
    manifest = {
        "version": VERSION, "status": "new_benchmark_spec_cpu_validated_model_regime_unmeasured",
        "target_parameters": PARAMETERS, "input_alphabet": INPUT_ALPHABET, "dsl_alphabet": DSL_ALPHABET,
        "length_range_inclusive": [0, MAX_LENGTH], "class_ids": CLASS_IDS,
        "class_rule": "nonoverlapping RB count in {0,1,2plus} crossed with endswith(R)",
        "empty_input_class": test_class(""), "empty_input_split": split_of(""),
        "per_suite_class_denominators": dict.fromkeys(CLASS_IDS, PER_CLASS),
        "tests_per_suite": PER_CLASS * len(CLASS_IDS), "empty_class_policy": "reject dataset/profile",
        "split_rule": "SHA256(version+'|input|'+tape) integer mod100: train<50; selection<70; heldout<85; else test",
        "same_split_input_reuse": True, "cross_split_input_reuse": False,
        "support_generation": "all RB tapes length<=12; all B^a R^b length<=64; 4096 seeded random length13..64; repeated RB paths",
        "support_count": len(support), "support_sha256": sha(support),
        "reference_usage": "CPU executable witness only; not Solver imitation data, hint, or model completion",
        "prompt_sha256": sha(prompt), "wrapper_sha256": hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
        "model_results_read": False, "independent_task_family_generalization": False,
        "split_rows": {}, "pool_counts": {},
    }
    files = {}
    for split, count in SPLIT_COUNTS.items():
        class_pools = {cls: pools[split, cls] for cls in CLASS_IDS}
        if any(len(pool) < PER_CLASS for pool in class_pools.values()):
            raise ValueError(f"insufficient class support: {split}")
        manifest["pool_counts"][split] = {cls: len(pool) for cls, pool in class_pools.items()}
        rows, used_suites = [], set()
        for idx in range(count):
            attempt = 0
            while True:
                # Fixed per-suite seed; retry is solely suite duplication handling.
                rng = random.Random(f"{VERSION}|suite|{split}|{idx}|{attempt}")
                tapes = [x for cls in CLASS_IDS for x in rng.sample(class_pools[cls], PER_CLASS)]
                # The unique empty input is explicitly covered once, in its hash-assigned split.
                if idx == 0 and split == split_of("") and "" not in tapes:
                    tapes[0] = ""
                key = tuple(sorted(tapes))
                if key not in used_suites:
                    used_suites.add(key)
                    break
                attempt += 1
            cases = [{"input": x, "expected_output": expected_output(x), "expected_accepted": True,
                      "check_output": True, "test_class": test_class(x),
                      "test_id": sha({"version": VERSION, "input": x}), "description": ""} for x in tapes]
            row = {"id": f"{VERSION}_{split}_{idx:04d}", "candidate_id": VERSION,
                   "messages": [{"role": "user", "content": prompt}], "ground_truth": cases,
                   "generator_config": canonical({"official_parameter": PARAMETERS,
                        "benchmark_version": VERSION, "input_alphabet": INPUT_ALPHABET, "split": split}),
                   "class_denominators": dict.fromkeys(CLASS_IDS, PER_CLASS), "hint": None}
            rows.append(row)
        path = out / f"target_{split}.jsonl"
        write_utf8(path, "".join(canonical(row) + "\n" for row in rows))
        files[split] = path
        manifest["split_rows"][split] = {"count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "distinct_input_tapes": len({c['input'] for r in rows for c in r['ground_truth']})}
    write_utf8(out / "reference_oracle.manufactoria", reference_program())
    write_utf8(out / "target_prompt.txt", prompt)
    write_utf8(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_dataset(out, parser_path):
    parser = load_python(parser_path, "verge_benchmark_actual_vendor_parser")
    factory = parser.create_robot_factory(reference_program())
    rows = {split: [json.loads(line) for line in (out / f"target_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            for split in SPLIT_COUNTS}
    report = {"parser_path": str(parser_path), "parser_sha256": hashlib.sha256(parser_path.read_bytes()).hexdigest(),
              "python": sys.version, "mode": "CPU_reference_program_only_no_model_outputs",
              "reference_nodes": len(factory.nodes), "reference_sha256": sha(reference_program()),
              "split_checks": {}, "pairwise_input_intersection_sizes": {}, "all_pass": True}
    used = {}
    for split, split_rows in rows.items():
        used[split] = set()
        cases, max_steps = 0, 0
        for row in split_rows:
            assert row["hint"] is None
            assert len(row["ground_truth"]) == PER_CLASS * len(CLASS_IDS)
            assert Counter(c["test_class"] for c in row["ground_truth"]) == Counter(dict.fromkeys(CLASS_IDS, PER_CLASS))
            assert len({c["input"] for c in row["ground_truth"]}) == len(row["ground_truth"])
            for case in row["ground_truth"]:
                tape = case["input"]
                assert split_of(tape) == split and test_class(tape) == case["test_class"]
                assert case["expected_output"] == expected_output(tape)
                actual = factory.process_robot(tape)
                assert actual.finished and actual.final_tape == case["expected_output"], (split, tape, actual)
                used[split].add(tape)
                cases += 1
                max_steps = max(max_steps, len(actual.path))
        report["split_checks"][split] = {"rows": len(split_rows), "reference_exact_passes": cases,
                                         "tests": cases, "max_execution_steps": max_steps,
                                         "unique_input_tapes": len(used[split])}
    for a, b in itertools.combinations(rows, 2):
        n = len(used[a] & used[b])
        assert n == 0
        report["pairwise_input_intersection_sizes"][a + "/" + b] = n
    exhaustive = ["".join(bits) for n in range(15) for bits in itertools.product(INPUT_ALPHABET, repeat=n)]
    additional = set(input_support()) - set(exhaustive)
    maximum, repeated_node_cases = 0, 0
    for tape in exhaustive + sorted(additional):
        result = factory.process_robot(tape)
        assert result.finished and result.final_tape == expected_output(tape), (tape, result)
        maximum = max(maximum, len(result.path))
        repeated_node_cases += int(len(set(result.path)) < len(result.path))
    report["exhaustive_reference_check"] = {"lengths": [0, 14], "cases": len(exhaustive), "all_pass": True}
    report["additional_reference_check"] = {"cases": len(additional), "maximum_input_length": MAX_LENGTH,
                                            "all_pass": True}
    report["max_execution_steps"] = maximum
    report["cases_with_actually_revisited_nodes"] = repeated_node_cases
    # Negative controls establish both scope and nonvacuous verifier behavior.
    wrong_prefix = parser.create_robot_factory(reference_program("RBYGRR"))
    negative_cases = ["", "B", "R", "RB", "RBR", "RBRB", "RBRBR"]
    report["wrong_prefix_control"] = [int((r := wrong_prefix.process_robot(x)).finished and r.final_tape == expected_output(x)) for x in negative_cases]
    assert not any(report["wrong_prefix_control"])
    raw_four_color = factory.process_robot("Y")
    report["outside_domain_original_four_color_Y"] = {"finished": raw_four_color.finished,
        "actual_output": raw_four_color.final_tape, "original_expected": PREFIX + "Y",
        "exact_pass": bool(raw_four_color.finished and raw_four_color.final_tape == PREFIX + "Y")}
    assert not report["outside_domain_original_four_color_Y"]["exact_pass"]
    write_utf8(out / "cpu_reference_verification.json", json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "generated")
    cli.add_argument("--wrapper", type=Path)
    cli.add_argument("--parser", type=Path)
    args = cli.parse_args()
    if args.wrapper:
        print(json.dumps({"prepared": prepare(args.output, args.wrapper)}, indent=2))
    if args.parser:
        print(json.dumps({"verified": verify_dataset(args.output, args.parser)}, indent=2))
    if not args.wrapper and not args.parser:
        cli.error("provide --wrapper to generate and/or --parser to verify")
