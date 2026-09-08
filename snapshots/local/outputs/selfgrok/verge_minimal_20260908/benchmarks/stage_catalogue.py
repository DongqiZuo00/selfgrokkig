"""Frozen, simple stage candidates for a bounded VERGE learnability screen.

No GPU submission or model sampling. Reference DSL is a CPU verification oracle,
kept separate from model rows and never supplied as Solver supervision.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1] / "verge_operational_20260908"
VERSION = "verge_ignition_stage_catalogue_v1_20260908"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TARGET = load_module(RELEASE / "benchmark/benchmark.py", "verge_ignition_frozen_target")
STAGES = load_module(RELEASE / "stages/operational_stages.py", "verge_ignition_stage_identity")

DEFINITIONS = (
    {"id": "identity", "operation": "identity", "max_length": 8, "kind": "tape_transform"},
    {"id": "append_R", "operation": "append", "literal": "R", "max_length": 8, "kind": "tape_transform"},
    {"id": "append_B", "operation": "append", "literal": "B", "max_length": 8, "kind": "tape_transform"},
    {"id": "replace_R_to_B", "operation": "replace", "old": "R", "new": "B", "max_length": 8, "kind": "tape_transform"},
    {"id": "replace_RB_to_YG", "operation": "replace", "old": "RB", "new": "YG", "max_length": 8, "kind": "registered"},
    {"id": "prepend_R", "operation": "prepend", "literal": "R", "max_length": 8, "kind": "tape_transform"},
    {"id": "prepend_RB", "operation": "prepend", "literal": "RB", "max_length": 8, "kind": "tape_transform"},
    {"id": "target_small2", "operation": "fixed_target", "max_length": 2, "kind": "registered"},
    {"id": "target_small3", "operation": "fixed_target", "max_length": 3, "kind": "registered"},
)
DEFINITION_BY_ID = {d["id"]: d for d in DEFINITIONS}
VARIANTS = ("compact", "official_syntax")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def target_definition(spec):
    # Identity excludes stage row id / split / wording variant. All parameters
    # defining the task and its input domain remain visible before hashing.
    task = {k: v for k, v in spec.items() if k not in {"id", "kind"}}
    task.update(alphabet="RB", min_length=0, output_alphabet="RBYG")
    if spec["operation"] == "fixed_target":
        task["parameters"] = TARGET.PARAMETERS
    return task


def expectation(spec, tape):
    if not isinstance(tape, str) or set(tape) - set("RB") or not 0 <= len(tape) <= spec["max_length"]:
        raise ValueError("input outside frozen stage domain")
    op = spec["operation"]
    if op == "identity":
        return tape
    if op == "append":
        return tape + spec["literal"]
    if op == "prepend":
        return spec["literal"] + tape
    if op == "replace":
        return tape.replace(spec["old"], spec["new"])
    if op == "fixed_target":
        return TARGET.expected_output(tape)
    raise ValueError("unknown registered stage operation")


def stage_inputs(spec):
    available = ["".join(chars) for n in range(spec["max_length"] + 1)
                 for chars in itertools.product("RB", repeat=n)]
    pool = sorted((x for x in available if TARGET.split_of(x) == "train"), key=lambda x: (len(x), x))
    if len(pool) <= 16:
        result = pool
    else:
        # Include every permitted short tape, then fixed seeded longer cases.
        result = [x for x in pool if len(x) <= 3]
        long_pool = [x for x in pool if x not in result]
        rng = random.Random(VERSION + "|suite|" + str(spec["max_length"]))
        result += rng.sample(long_pool, 16 - len(result))
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("stage requires at least two distinct train-side inputs")
    return result


def expression(spec):
    inp = {"op": "input"}
    op = spec["operation"]
    if op == "identity":
        return inp
    if op == "append":
        return {"op": "concat", "args": [inp, {"op": "literal", "value": spec["literal"]}]}
    if op == "prepend":
        return {"op": "concat", "args": [{"op": "literal", "value": spec["literal"]}, inp]}
    if op == "replace" and len(spec["old"]) == 1:
        return {"op": "replace", "arg": inp, "old": spec["old"], "new": spec["new"]}
    return None


def task_description(spec):
    base = f"The input tape contains only R and B, with length from 0 to {spec['max_length']}. "
    op = spec["operation"]
    if op == "identity":
        task = "Leave the tape unchanged."
    elif op == "append":
        task = f"Append exactly one {spec['literal']} at the end of the tape. Keep the original symbols in their original order."
    elif op == "prepend":
        task = f"Prepend {spec['literal']} at the beginning of the tape. Keep the original symbols in their original order."
    elif op == "replace":
        task = f"Replace every nonoverlapping occurrence of {spec['old']} with {spec['new']}. Keep every other symbol in its original order."
    else:
        task = "Replace every nonoverlapping occurrence of RB with YG, then prepend RBYGRB. Keep all other symbols in order."
    return base + task + " All valid inputs must reach end with exactly the required output. The full standard R/B/Y/G DSL is available."


COMPACT_SYNTAX = """Write a Manufactoria program. Output one ```manufactoria ...``` code block.
The tape is a FIFO sequence. A painter appends its color to the BACK of the tape.
PULLER_RB removes the FRONT symbol only if it is R or B, then follows its [R] or [B] route.
Its [EMPTY] route leaves the tape unchanged when empty or when the front is Y/G.
PULLER_YG does the same for Y/G; its [EMPTY] route also leaves an R/B front unchanged.
Reaching end accepts. NONE, an undefined route, or 1000 execution steps rejects.
Available painter types: PAINTER_RED, PAINTER_BLUE, PAINTER_YELLOW, PAINTER_GREEN.
Node and route syntax (angle brackets denote placeholders, not literal code):
START start:
    NEXT <node_id>
PULLER_RB <node_id>:
    [R] <node_id>
    [B] <node_id>
    [EMPTY] <node_id>
PULLER_YG <node_id>:
    [Y] <node_id>
    [G] <node_id>
    [EMPTY] <node_id>
<painter_type> <node_id>:
    NEXT <node_id>
END end
Use unique node ids and four-space route indentation. END has no colon.
Task:
"""


def prompt(spec, variant):
    variant = {"concise": "compact", "official": "official_syntax"}.get(variant, variant)
    if variant == "compact":
        syntax = COMPACT_SYNTAX
    elif variant == "official_syntax":
        original = (RELEASE / "benchmark/generated/target_prompt.txt").read_text(encoding="utf-8")
        if "# Task" not in original:
            raise ValueError("frozen official wrapper prompt is missing its task boundary")
        syntax = original.split("# Task", 1)[0] + "# Task\n"
    else:
        raise ValueError("unknown wording variant")
    return syntax + task_description(spec)


def build_rows(variant="compact"):
    variant = {"concise": "compact", "official": "official_syntax"}.get(variant, variant)
    rows = []
    for spec in DEFINITIONS:
        inputs = stage_inputs(spec)
        definition = target_definition(spec)
        identity = STAGES.instance_tuple_from_manifest(definition, inputs)
        cases = [{"input": tape, "expected_output": expectation(spec, tape), "expected_accepted": True,
                  "check_output": True, "test_id": digest({"definition": definition, "input": tape})} for tape in inputs]
        if len({c["expected_output"] for c in cases}) < 2:
            raise ValueError("constant-output stage must not enter catalogue")
        rows.append({"id": f"{VERSION}__{spec['id']}__{variant}", "candidate_id": spec["id"],
            "messages": [{"role": "user", "content": prompt(spec, variant)}], "ground_truth": cases, "hint": None,
            "prompt_variant": variant, "stage_kind": spec["kind"], "task_definition": definition,
            "instance_tuple": list(identity), "generator_config": canonical({"catalogue": VERSION,
                "task_definition": definition, "instance_tuple": identity, "source_partition": "train",
                "partition_rule": TARGET.VERSION + ":input_sha256_split", "prompt_variant": variant}),
            "restricted_expression": expression(spec), "solver_reward": "binary_full_pass"})
    return rows


def make_stage_rows(stage_id, prompt_variant="concise"):
    """Model-facing interface: list containing one frozen row, no oracle text.

    Each row can be independently sampled eight times for a binary reward group.
    Both concise/official and compact/official_syntax aliases are accepted.
    """
    if stage_id not in DEFINITION_BY_ID:
        raise KeyError(stage_id)
    return [row for row in build_rows(prompt_variant) if row["candidate_id"] == stage_id]


def stage_spec(stage_id):
    definition = DEFINITION_BY_ID[stage_id]
    row = make_stage_rows(stage_id)[0]
    base = {"kind": definition["kind"], "distribution": {"alphabet": "RB", "min_length": 0,
            "max_length": definition["max_length"], "corner_cases": [c["input"] for c in row["ground_truth"]]},
            "instance_tuples": [row["instance_tuple"]]}
    if definition["kind"] == "tape_transform":
        base["expression"] = expression(definition)
    else:
        base.update(family="verge_catalogue_" + stage_id, tier=0, mutation_tier=0)
    return base


def registered_families():
    return {"verge_catalogue_" + d["id"]: STAGES.RegisteredFamily(
        lambda tape, _spec, definition=d: expectation(definition, tape), VERSION + ":" + d["id"])
        for d in DEFINITIONS if d["kind"] == "registered"}


def reference_program(stage_id):
    """CPU oracle only, separate from build_rows/model-facing prompt artifacts."""
    spec = DEFINITION_BY_ID[stage_id]
    colors = {"R": "RED", "B": "BLUE", "Y": "YELLOW", "G": "GREEN"}
    if spec["operation"] == "identity":
        return "START start:\n    NEXT end\nEND end\n"
    if spec["operation"] == "append":
        return f"START start:\n    NEXT paint\nPAINTER_{colors[spec['literal']]} paint:\n    NEXT end\nEND end\n"
    if spec["operation"] == "fixed_target":
        return TARGET.reference_program()
    prefix = spec.get("literal", "") if spec["operation"] == "prepend" else ""
    lines = ["START start:", "    NEXT marker", "PAINTER_YELLOW marker:", f"    NEXT {'prefix_0' if prefix else 'scan'}"]
    for i, color in enumerate(prefix):
        lines += [f"PAINTER_{colors[color]} prefix_{i}:", f"    NEXT {'scan' if i == len(prefix)-1 else 'prefix_'+str(i+1)}"]
    if spec["operation"] == "replace" and spec["old"] == "RB":
        lines += ["PULLER_RB scan:", "    [R] pending", "    [B] emit_b", "    [EMPTY] finish",
            "PAINTER_BLUE emit_b:", "    NEXT scan", "PULLER_RB pending:", "    [R] emit_r_pending", "    [B] emit_y", "    [EMPTY] finish_pending",
            "PAINTER_RED emit_r_pending:", "    NEXT pending", "PAINTER_YELLOW emit_y:", "    NEXT emit_g",
            "PAINTER_GREEN emit_g:", "    NEXT scan", "PULLER_YG finish_pending:", "    [Y] emit_last_r", "    [G] NONE", "    [EMPTY] NONE",
            "PAINTER_RED emit_last_r:", "    NEXT end"]
    else:
        r_emit = "emit_b" if spec["operation"] == "replace" else "emit_r"
        lines += ["PULLER_RB scan:", f"    [R] {r_emit}", "    [B] emit_b", "    [EMPTY] finish",
            "PAINTER_BLUE emit_b:", "    NEXT scan"]
        if r_emit == "emit_r":
            lines += ["PAINTER_RED emit_r:", "    NEXT scan"]
    lines += ["PULLER_YG finish:", "    [Y] end", "    [G] NONE", "    [EMPTY] NONE", "END end", ""]
    return "\n".join(lines)


def evaluate_stage(completion, row, create_factory):
    """Model screen interface: current prompt task's strict binary full pass."""
    spec = DEFINITION_BY_ID[row["candidate_id"]]
    cases = row["ground_truth"]
    if not cases or len({c["input"] for c in cases}) != len(cases):
        raise ValueError("invalid/empty stage suite")
    for case in cases:
        if case.get("expected_output") != expectation(spec, case["input"]) or case.get("expected_accepted") is not True or case.get("check_output") is not True:
            raise ValueError("stage case differs from frozen catalogue definition")
    identity = STAGES.instance_tuple_from_manifest(row["task_definition"], [c["input"] for c in cases])
    if row["instance_tuple"] != list(identity) or row["task_definition"] != target_definition(spec):
        raise ValueError("stage task/suite identity mismatch")
    try:
        factory = create_factory(TARGET.extract_program(completion))
        parsed, parse_error = True, None
    except Exception as error:
        factory, parsed, parse_error = None, False, str(error)
    per_test = []
    for case in cases:
        passed, reason, steps = 0, parse_error, 0
        if factory is not None:
            try:
                result = factory.process_robot(case["input"])
                passed = int(result.finished and result.final_tape == case["expected_output"])
                reason = result.rejection_reason if not result.finished else (None if passed else "output_mismatch")
                steps = len(result.path)
            except Exception as error:
                reason = str(error)
        per_test.append({"test_id": case["test_id"], "input": case["input"], "pass": passed, "reason": reason, "steps": steps})
    return {"parse_valid": parsed, "reward": int(all(x["pass"] for x in per_test)),
            "passed_cases": sum(x["pass"] for x in per_test), "total_cases": len(per_test), "per_test": per_test}


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    protected_inputs = set()
    target_manifest = json.loads((RELEASE / "benchmark/generated/manifest.json").read_text(encoding="utf-8"))
    protected_suites = []
    for split in ("selection", "heldout", "test"):
        for line in (RELEASE / "benchmark/generated" / f"target_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            inputs = [c["input"] for c in row["ground_truth"]]
            protected_inputs.update(inputs)
            protected_suites.append({"split": split, "inputs_sha256": STAGES.instance_tuple_from_manifest({"parameters": TARGET.PARAMETERS}, inputs)[1]})
    manifest = {"version": VERSION, "status": "candidate_stages_CPU_oracles_only_model_ignition_unmeasured",
        "fixed_target_parameters": TARGET.PARAMETERS, "fixed_target_version": TARGET.VERSION,
        "fixed_target_manifest_sha256": file_digest(RELEASE / "benchmark/generated/manifest.json"),
        "old_release_modified": False, "model_results_read_to_choose_tasks": False,
        "reference_is_training_supervision": False, "variants": list(VARIANTS), "candidates": [],
        "source_sha256": {str(p): file_digest(p) for p in (Path(TARGET.__file__), Path(STAGES.__file__), RELEASE / "benchmark/generated/target_prompt.txt")},
        "protected_actual_input_count": len(protected_inputs), "frozen_target_split_sha256": target_manifest["split_rows"]}
    for variant in VARIANTS:
        rows = build_rows(variant)
        write(output / f"screening_rows_{variant}.jsonl", "".join(canonical(r) + "\n" for r in rows))
        for row in rows:
            assert not set(c["input"] for c in row["ground_truth"]) & protected_inputs
            assert all(TARGET.split_of(c["input"]) == "train" for c in row["ground_truth"])
            if variant == VARIANTS[0]:
                spec = DEFINITION_BY_ID[row["candidate_id"]]
                manifest["candidates"].append({"stage_id": row["candidate_id"], "kind": spec["kind"],
                    "task_definition": row["task_definition"], "instance_tuple": row["instance_tuple"],
                    "case_count": len(row["ground_truth"]), "distinct_expected_outputs": len({c["expected_output"] for c in row["ground_truth"]}),
                    "inputs": [c["input"] for c in row["ground_truth"]],
                    "inputs_containing_RB": sum("RB" in c["input"] for c in row["ground_truth"]),
                    "empty_input_in_train_suite": any(c["input"] == "" for c in row["ground_truth"]),
                    "protected_input_overlap": 0, "reference_file": f"references/{spec['id']}.manufactoria"})
                write(output / "references" / f"{spec['id']}.manufactoria", reference_program(spec["id"]))
    manifest["row_file_sha256"] = {variant: file_digest(output / f"screening_rows_{variant}.jsonl") for variant in VARIANTS}
    write(output / "catalogue_manifest.json", json.dumps(manifest, indent=2) + "\n")
    catalogue = {"version": VERSION, "model_rows_have_reference_program": False,
        "stages": [{"stage_id": d["id"], "kind": d["kind"], "spec": stage_spec(d["id"]),
            "rows": make_stage_rows(d["id"], "concise"),
            "rows_by_variant": {v: make_stage_rows(d["id"], v) for v in ("concise", "official")}}
            for d in DEFINITIONS]}
    write(output / "catalogue.json", json.dumps(catalogue, indent=2) + "\n")
    return manifest


def verify(output, parser_path):
    parser = load_module(parser_path, "verge_ignition_real_parser")
    report = {"mode": "CPU_reference_only_no_model", "parser_path": str(parser_path),
              "parser_sha256": file_digest(parser_path), "candidates": [], "all_pass": True}
    for row in build_rows():
        spec = DEFINITION_BY_ID[row["candidate_id"]]
        program = reference_program(spec["id"])
        evaluation = evaluate_stage(program, row, parser.create_robot_factory)
        assert evaluation["reward"] == 1
        factory = parser.create_robot_factory(program)
        all_inputs = ["".join(chars) for n in range(spec["max_length"] + 1) for chars in itertools.product("RB", repeat=n)]
        maximum_steps = 0
        for tape in all_inputs:
            result = factory.process_robot(tape)
            assert result.finished and result.final_tape == expectation(spec, tape), (spec["id"], tape, result)
            maximum_steps = max(maximum_steps, len(result.path))
        report["candidates"].append({"stage_id": spec["id"], "training_suite": evaluation,
            "oracle_exhaustive_cases": len(all_inputs), "exhaustive_all_pass": True, "max_execution_steps": maximum_steps,
            "program_nodes": len(factory.nodes), "model_rollouts": 0})
    report["total_training_suite_tests"] = sum(x["training_suite"]["total_cases"] for x in report["candidates"])
    report["total_exhaustive_oracle_tests"] = sum(x["oracle_exhaustive_cases"] for x in report["candidates"])
    write(output / "cpu_oracle_verification.json", json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=HERE / "generated")
    cli.add_argument("--parser", type=Path)
    args = cli.parse_args()
    manifest = prepare(args.output)
    print(json.dumps({"prepared_candidates": len(manifest["candidates"]), "variants": list(VARIANTS), "output": str(args.output)}, indent=2))
    if args.parser:
        print(json.dumps(verify(args.output, args.parser), indent=2))
