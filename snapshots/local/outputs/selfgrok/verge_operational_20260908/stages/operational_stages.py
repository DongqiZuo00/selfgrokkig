"""Explicit stage contracts and cumulative Challenger positive-example buffer.

This is a new operational specification, not a claim about prior implementation
or model performance. Only trusted, registered Python functions may run; proposed
transforms are interpreted as a bounded data-only DSL, never eval/exec.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def instance_tuple_from_manifest(target_definition: Mapping, suite_inputs: Sequence[str]) -> tuple[str, str]:
    """Trusted manifest adapter: identity excludes split names and row IDs."""
    if not target_definition or any(k in target_definition for k in ("split", "id", "instance_id", "split_name")):
        raise ValueError("target definition must contain task parameters, not split/id labels")
    if not suite_inputs or any(not isinstance(x, str) for x in suite_inputs):
        raise ValueError("suite must contain concrete input tapes")
    return _digest(target_definition), _digest(sorted(set(suite_inputs)))


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def validate_distribution(distribution: Mapping[str, Any]) -> dict:
    required = {"alphabet", "min_length", "max_length", "corner_cases"}
    if set(distribution) != required:
        raise ValueError(f"distribution fields must be {sorted(required)}")
    alphabet = distribution["alphabet"]
    if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet must be a nonempty string of unique symbols")
    if len(alphabet) > 64:
        raise ValueError("alphabet limit is 64 symbols")
    lo = _integer(distribution["min_length"], "min_length", 0, 1024)
    hi = _integer(distribution["max_length"], "max_length", lo, 1024)
    corners = distribution["corner_cases"]
    if not isinstance(corners, list) or len(corners) < 2 or len(corners) > 128:
        raise ValueError("provide 2..128 explicit corner cases")
    for tape in corners:
        if not isinstance(tape, str) or not lo <= len(tape) <= hi or not set(tape) <= set(alphabet):
            raise ValueError("corner case outside declared alphabet/length distribution")
    if len(set(corners)) < 2:
        raise ValueError("corner cases must contain at least two distinct inputs")
    return _copy(distribution)


def generate_cases(distribution: Mapping[str, Any], seed: int = 0) -> list[str]:
    """Frozen protocol verge-stage-cases-v1: corners, length edges, 32 draws."""
    d = validate_distribution(distribution)
    rng = random.Random(_integer(seed, "seed", 0, 2**63 - 1))
    cases = list(d["corner_cases"])
    for n in (d["min_length"], d["max_length"]):
        cases.extend(symbol * n for symbol in d["alphabet"])
    for _ in range(32):
        n = rng.randint(d["min_length"], d["max_length"])
        cases.append("".join(rng.choice(d["alphabet"]) for _ in range(n)))
    return list(dict.fromkeys(cases))


def evaluate_transform(expression: Mapping[str, Any], tape: str) -> str | bool:
    """Finite DSL: input/literal/reverse/rotate/slice/replace/concat/contains/equals.

    Max depth 8, max 64 nodes, strings max 4096 characters; no loops, imports,
    file/network operations, attributes, or arbitrary calls exist in the grammar.
    """
    if not isinstance(tape, str) or len(tape) > 1024:
        raise ValueError("input tape must be a string of at most 1024 characters")
    nodes = 0

    def string(value: Any) -> str:
        if type(value) is not str or len(value) > 4096:
            raise ValueError("DSL string type/output limit violated")
        return value

    def visit(node: Any, depth: int) -> str | bool:
        nonlocal nodes
        nodes += 1
        if nodes > 64 or depth > 8 or not isinstance(node, dict):
            raise ValueError("DSL node/depth/type limit violated")
        op = node.get("op")
        fields = {
            "input": {"op"}, "literal": {"op", "value"},
            "reverse": {"op", "arg"}, "rotate": {"op", "arg", "offset"},
            "slice": {"op", "arg", "start", "stop"},
            "replace": {"op", "arg", "old", "new"},
            "concat": {"op", "args"}, "contains": {"op", "arg", "value"},
            "equals": {"op", "left", "right"},
        }
        if op not in fields or set(node) != fields[op]:
            raise ValueError("unknown DSL operation or fields")
        if op == "input":
            return tape
        if op == "literal":
            return string(node["value"])
        if op == "concat":
            args = node["args"]
            if not isinstance(args, list) or not 1 <= len(args) <= 4:
                raise ValueError("concat requires 1..4 operands")
            return string("".join(string(visit(arg, depth + 1)) for arg in args))
        if op == "equals":
            return string(visit(node["left"], depth + 1)) == string(visit(node["right"], depth + 1))
        arg = string(visit(node["arg"], depth + 1))
        if op == "reverse":
            return arg[::-1]
        if op == "rotate":
            offset = _integer(node["offset"], "offset", -1024, 1024)
            offset = offset % len(arg) if arg else 0
            return arg[offset:] + arg[:offset]
        if op == "slice":
            start = _integer(node["start"], "start", -4096, 4096)
            stop = _integer(node["stop"], "stop", -4096, 4096)
            return arg[start:stop]
        if op == "contains":
            return string(node["value"]) in arg
        old, new = string(node["old"]), string(node["new"])
        if len(old) != 1 or len(new) > 1:
            raise ValueError("replace permits one symbol to zero/one symbol only")
        return arg.replace(old, new)

    return visit(expression, 0)


@dataclass(frozen=True)
class RegisteredFamily:
    expected: Callable[[str, Mapping[str, Any]], str | bool]
    protocol_id: str
    tiers: tuple[int, ...] = (0, 1, 2, 3)
    mutation_tiers: tuple[int, ...] = (0, 1, 2, 3)
    related_hint_families: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedProgram:
    entry_id: str
    program: str
    program_sha256: str
    family: str
    instance_size: int
    source_instance_tuple: tuple
    source_split: str
    verification_id: str
    verifier_protocol_id: str
    verification_test_ids: tuple[str, ...]


class VerifiedProgramLibrary:
    """Library insertion consumes recorded full-pass verifier evidence.

    It does not execute the supplied program. The evidence id must resolve to
    the trusted external verifier log in a real run; synthetic ids are fixtures.
    """
    def __init__(self) -> None:
        self.entries: dict[str, VerifiedProgram] = {}

    def add(self, *, entry_id: str, program: str, family: str, instance_size: int,
            source_instance_tuple: Sequence, source_split: str,
            verification_id: str, verifier_protocol_id: str,
            verification_test_ids: Sequence[str], exact_pass: Sequence[int]) -> VerifiedProgram:
        if entry_id in self.entries:
            raise ValueError("verified-program entry id already exists")
        if source_split != "train":
            raise ValueError("hint programs must originate from train instances")
        if not all(isinstance(x, str) and x for x in (entry_id, program, family, verification_id, verifier_protocol_id)):
            raise ValueError("program and provenance identifiers are required")
        _integer(instance_size, "instance_size", 0, 1024)
        params = _parameter_tuple(source_instance_tuple)
        ids, passes = tuple(verification_test_ids), list(exact_pass)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("verification test ids must be nonempty and unique")
        if len(passes) != len(ids) or any(type(x) is not int or x != 1 for x in passes):
            raise ValueError("every recorded verifier test must exact-pass")
        item = VerifiedProgram(entry_id, program, hashlib.sha256(program.encode()).hexdigest(),
                               family, instance_size, params, source_split, verification_id,
                               verifier_protocol_id, ids)
        self.entries[entry_id] = item
        return item


def _parameter_tuple(value: Any) -> tuple:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
        or any(not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in value)):
        raise ValueError("instance identity must be (target-definition SHA256, suite-input-set SHA256), excluding split/id labels")
    return tuple(value)


@dataclass(frozen=True)
class HintProbe:
    with_hint: Sequence
    without_hint: Sequence
    instance_ids: Sequence[str]
    rollout_seed_ids: Sequence
    condition_id: str


def _probe_report(probe: HintProbe, delta: float) -> dict:
    if len(probe.instance_ids) != 32 or len(set(probe.instance_ids)) != 32 or not probe.condition_id:
        raise ValueError("hint probe requires 32 paired instance ids and one condition id")
    def check(array: Sequence, *, seeds: bool = False) -> None:
        if len(array) != 32:
            raise ValueError("each hint probe arm must have shape [32 instances][8 rollouts]")
        for group in array:
            if len(group) != 8:
                raise ValueError("each hint probe arm must have shape [32][8]")
            if seeds:
                if any(type(x) is not int or x < 0 for x in group) or len(set(group)) != 8:
                    raise ValueError("paired rollout seeds must be distinct nonnegative integers per instance")
            elif any(type(x) is not int or x not in (0, 1) for x in group):
                raise ValueError("hint probe outcomes must be binary integers")
    check(probe.with_hint)
    check(probe.without_hint)
    check(probe.rollout_seed_ids, seeds=True)
    p_w = sum(map(sum, probe.with_hint)) / 256
    p_wo = sum(map(sum, probe.without_hint)) / 256
    return {"accepted": p_w > p_wo and p_wo < 1 - delta,
            "p_with_hint": p_w, "p_without_hint": p_wo,
            "improves": p_w > p_wo, "unsaturated": p_wo < 1 - delta,
            "mixed_groups_with_hint": sum(0 < sum(g) < 8 for g in probe.with_hint),
            "mixed_groups_without_hint": sum(0 < sum(g) < 8 for g in probe.without_hint),
            "shape_per_arm": [32, 8], "rollouts_per_arm": 256,
            "total_rollouts": 512, "condition_id": probe.condition_id,
            "mixed_groups_are_diagnostic_only": True}


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    checks: dict
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


def validate_stage(spec: Mapping[str, Any], *, registry: Mapping[str, RegisteredFamily],
                   protected_tuples: Sequence | None = None, library: VerifiedProgramLibrary | None = None,
                   hint_probe: HintProbe | None = None, delta: float = 0.1) -> ValidationResult:
    """Exactly four gates. A non-hint stage has explicit hint_regret='NA'.

    protected_tuples must be the union of selection/held-out/test manifests,
    never inferred from model outcomes. Caller must freeze and persist it.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between 0 and 1")
    spec = dict(spec)
    alias_used = spec.get("kind") == "restricted_transform"
    if alias_used:
        spec["kind"] = "tape_transform"
    checks = {"executable": False, "nonconstant": False, "no_leakage": False,
              "hint_regret": "NA" if spec.get("kind") != "hinted_target" else False}
    details: dict = {"implementation": "verge-operational-20260908", "delta": delta,
                     "canonical_kind": spec.get("kind"), "legacy_alias_normalized": alias_used}
    protected: set[tuple] = set()
    try:
        if protected_tuples is None:
            raise ValueError("explicit selection/held-out/test manifest tuples are required")
        protected = {_parameter_tuple(x) for x in protected_tuples}
        instances = [_parameter_tuple(x) for x in spec.get("instance_tuples", [])]
        if not instances:
            raise ValueError("stage requires explicit instantiated parameter tuples")
        overlap = [x for x in instances if x in protected]
        checks["no_leakage"] = not overlap
        details["leakage"] = {"protected_tuple_count": len(protected), "instance_count": len(instances),
                              "overlap_count": len(overlap), "protected_manifest_sha256": _digest(sorted(map(list, protected), key=str))}
    except (TypeError, ValueError) as error:
        details["leakage_error"] = str(error)
    try:
        kind = spec["kind"]
        common = {"kind", "distribution", "instance_tuples"}
        allowed = {
            "registered": common | {"family", "tier", "mutation_tier"},
            "tape_transform": common | {"expression"},
            "hinted_target": common | {"family", "tier", "mutation_tier", "target_size", "hint_id", "hint_relation"},
        }
        if kind not in allowed or set(spec) != allowed[kind]:
            raise ValueError("unknown stage kind or unexpected/missing fields")
        d = validate_distribution(spec["distribution"])
        if kind == "tape_transform":
            expected = lambda tape: evaluate_transform(spec["expression"], tape)
            details["protocol_id"] = "verge-stage-cases-v1"
        else:
            family = registry[spec["family"]]
            if not family.protocol_id:
                raise ValueError("registered family must name its frozen verifier/test protocol")
            tier = _integer(spec["tier"], "tier", 0, 64)
            mutation = _integer(spec["mutation_tier"], "mutation_tier", 0, 16)
            if tier not in family.tiers or mutation not in family.mutation_tiers:
                raise ValueError("tier or mutation tier is not registered")
            # Hint metadata is prompt-only and is never passed to the verifier.
            verifier_spec = {k: spec[k] for k in ("family", "tier", "mutation_tier", "distribution")}
            expected = lambda tape: family.expected(tape, verifier_spec)
            details["protocol_id"] = family.protocol_id
        if kind == "hinted_target":
            size = _integer(spec["target_size"], "target_size", 0, 1024)
            if library is None or spec["hint_id"] not in library.entries:
                raise ValueError("hint must reference a verified program library entry")
            hint = library.entries[spec["hint_id"]]
            if hint.source_split != "train":
                raise ValueError("hint must originate from a train instance")
            if spec["hint_relation"] not in ("related_train", "smaller_train"):
                raise ValueError("hint relation must be explicitly registered")
            if spec["hint_relation"] == "smaller_train":
                if hint.family != spec["family"] or hint.instance_size >= size:
                    raise ValueError("smaller hint must be same-family and strictly smaller")
            elif hint.family not in (spec["family"], *family.related_hint_families):
                raise ValueError("cross-family related hint requires a frozen family-relation declaration")
            if hint.source_instance_tuple in protected:
                checks["no_leakage"] = False
                raise ValueError("hint library source overlaps a protected split")
            details["hint_provenance"] = asdict(hint)
        outputs = [expected(tape) for tape in generate_cases(d)]
        if any(type(x) not in (str, bool) or (type(x) is str and len(x) > 4096) for x in outputs):
            raise ValueError("expected function must return a bounded string or binary predicate")
        if len({type(x) for x in outputs}) != 1:
            raise ValueError("expected function output type must be consistent")
        checks["executable"] = True
        corner_outputs = [expected(tape) for tape in d["corner_cases"]]
        checks["nonconstant"] = len(set(corner_outputs)) >= 2
        details["case_count"] = len(outputs)
        details["distinct_corner_outputs"] = len(set(corner_outputs))
        if kind == "hinted_target":
            if hint_probe is None:
                raise ValueError("hint stage requires paired hint-regret probe evidence")
            details["hint_probe"] = _probe_report(hint_probe, delta)
            checks["hint_regret"] = details["hint_probe"]["accepted"]
        elif hint_probe is not None:
            raise ValueError("non-hint stage must not be supplied fabricated hint-regret arms")
    except Exception as error:
        details["validation_error"] = str(error)
        # Metadata/probe errors must not leave an accepted stage behind.
        if checks["hint_regret"] == "NA":
            checks["executable"] = False
    return ValidationResult(all(x is True or x == "NA" for x in checks.values()), checks, details)


HINT_START = "<VERGE_TRAIN_HINT>"
HINT_END = "</VERGE_TRAIN_HINT>"


def render_target_prompt(base_prompt: str, *, hint: VerifiedProgram | None = None,
                         final_evaluation: bool = False) -> str:
    """Final target probes/evaluation accept only the untouched base prompt."""
    if not isinstance(base_prompt, str) or HINT_START in base_prompt or HINT_END in base_prompt:
        raise ValueError("base prompt contains a reserved training-hint marker")
    if final_evaluation and hint is not None:
        raise ValueError("hints are forbidden in final target evaluation/probes")
    if hint is None:
        return base_prompt
    if HINT_START in hint.program or HINT_END in hint.program:
        raise ValueError("hint program contains a reserved marker")
    return f"{base_prompt}\n{HINT_START}\nVerified related training solution ({hint.entry_id}):\n{hint.program}\n{HINT_END}"


class RejectionSamplingBuffer:
    """Append all round experiences; train only cumulative, significant J+.

    plan_update returns a one-epoch training plan, never launches training.
    Only successful, evidence-linked completion consumes the new-example count.
    """
    def __init__(self, n_min: int) -> None:
        self.n_min = _integer(n_min, "n_min", 1, 10**9)
        self.archive: list[dict] = []
        self.examples: list[dict] = []
        self.trained_ids: set[str] = set()
        self.completed_updates: list[dict] = []
        self.failed_updates: list[dict] = []
        self._rounds: set[tuple] = set()
        self._active: dict | None = None
        self._plan_counter = 0

    @property
    def new_count(self) -> int:
        return sum(row["example_id"] not in self.trained_ids for row in self.examples)

    def add_round(self, *, round_id: str, seed: int, context: Mapping, base: Any,
                  candidates: Sequence[Mapping], j_plus: Sequence[str], provenance: Mapping) -> list[str]:
        _integer(seed, "seed", 0, 2**63 - 1)
        if not round_id or (seed, round_id) in self._rounds:
            raise ValueError("round id must be nonempty and unique within each seed")
        if any(not provenance.get(k) for k in ("run_id", "condition_id", "evidence_id")):
            raise ValueError("round provenance needs run_id, condition_id, evidence_id")
        ids = [c["branch_id"] for c in candidates]
        if len(ids) != len(set(ids)) or not set(j_plus) <= set(ids) or len(j_plus) != len(set(j_plus)):
            raise ValueError("candidate ids must be unique; J+ must be a unique subset")
        experience = _copy({"round_id": round_id, "seed": seed, "context": context, "base": base,
                            "candidates": candidates, "j_plus": j_plus, "provenance": provenance})
        additions = []
        for candidate in experience["candidates"]:
            if candidate["branch_id"] not in j_plus:
                continue
            gamma, ci = candidate["gamma"], candidate["paired_ci"]
            if (not isinstance(gamma, (float, int)) or isinstance(gamma, bool) or not math.isfinite(gamma)
                or len(ci) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in ci)
                or not (gamma > 0 and 0 < ci[0] <= ci[1])):
                raise ValueError("J+ requires positive Gamma and strictly positive paired CI")
            if "curriculum" not in candidate or not candidate.get("evidence_id"):
                raise ValueError("positive candidate needs curriculum and source evidence id")
            row = {"context": experience["context"], "base": experience["base"],
                   "curriculum": candidate["curriculum"], "seed": seed, "round_id": round_id,
                   "branch_id": candidate["branch_id"], "gamma": gamma, "paired_ci": ci,
                   "provenance": experience["provenance"], "candidate_evidence_id": candidate["evidence_id"]}
            row["example_id"] = _digest(row)
            additions.append(row)
        self.archive.append(experience)
        self.examples.extend(additions)
        self._rounds.add((seed, round_id))
        return [row["example_id"] for row in additions]

    def plan_update(self) -> dict | None:
        if self._active is not None or self.new_count < self.n_min:
            return None
        self._plan_counter += 1
        rows = _copy(self.examples)
        self._active = {"plan_id": f"challenger-rft-{self._plan_counter}", "epochs": 1,
                        "method": "rejection_sampling_fine_tuning", "cumulative_examples": rows,
                        "example_ids": [x["example_id"] for x in rows],
                        "new_count_at_plan": self.new_count, "dataset_sha256": _digest(rows)}
        return _copy(self._active)

    def complete_update(self, plan_id: str, *, success: bool, checkpoint_id: str | None = None,
                        training_log_id: str | None = None, failure_reason: str | None = None) -> None:
        if self._active is None or self._active["plan_id"] != plan_id:
            raise ValueError("unknown or already completed training plan")
        if success and (not checkpoint_id or not training_log_id):
            raise ValueError("successful training requires checkpoint and training log provenance")
        if not success and not failure_reason:
            raise ValueError("failed training must record its reason")
        event = {"plan_id": plan_id, "dataset_sha256": self._active["dataset_sha256"],
                 "checkpoint_id": checkpoint_id, "training_log_id": training_log_id,
                 "failure_reason": failure_reason}
        if success:
            self.trained_ids.update(self._active["example_ids"])
            self.completed_updates.append(event)
        else:
            self.failed_updates.append(event)
        self._active = None

    def to_dict(self) -> dict:
        return _copy({"n_min": self.n_min, "archive": self.archive, "examples": self.examples,
                      "trained_ids": sorted(self.trained_ids), "new_count": self.new_count,
                      "zero_positive_rounds": sum(not row["j_plus"] for row in self.archive),
                      "completed_updates": self.completed_updates, "failed_updates": self.failed_updates,
                      "active_plan": self._active})
