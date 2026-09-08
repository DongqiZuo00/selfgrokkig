from __future__ import annotations

import re
from typing import Any

from common import CREATE_ROBOT_FACTORY, extract_program


def required_prefix(row: dict[str, Any]) -> str:
    match = re.match(r"^Prepend\s+([BRYG]+)(?:\s|$)", str(row["name"]))
    if not match:
        raise ValueError(f"cannot recover PREPEND prefix from name: {row['name']!r}")
    return match.group(1)


def cumulative_conditions(completion: str, row: dict[str, Any]) -> list[int]:
    """Return the four distinct cumulative PREPEND rungs and full pass.

    The final cumulative condition and the official full-pass verdict coincide
    for this all-accepting PREPEND verifier, so both values are retained in the
    record while ``distinct_rungs`` remains four.
    """
    try:
        factory = CREATE_ROBOT_FACTORY(extract_program(completion))
    except Exception:
        return [0, 0, 0, 0, 0]

    prefix = required_prefix(row)
    all_finished = True
    all_prefixed = True
    all_residual = True
    full_pass = True
    for case in row["ground_truth"]:
        try:
            result = factory.process_robot(case.get("input", ""))
            expected = str(case.get("expected_output", ""))
            finished = bool(result.finished)
            prefixed = str(result.final_tape).startswith(prefix)
            residual = prefixed and str(result.final_tape)[len(prefix) :] == expected[len(prefix) :]
            passed = finished and str(result.final_tape) == expected
        except Exception:
            finished = prefixed = residual = passed = False
        all_finished = all_finished and finished
        all_prefixed = all_prefixed and prefixed
        all_residual = all_residual and residual
        full_pass = full_pass and passed

    a1 = True
    c1 = int(a1)
    c2 = int(a1 and all_finished)
    c3 = int(a1 and all_finished and all_prefixed)
    c4 = int(a1 and all_finished and all_prefixed and all_residual)
    c5 = int(full_pass)
    if c5 > c4:
        raise AssertionError("full pass must imply every cumulative PREPEND condition")
    return [c1, c2, c3, c4, c5]


def condition_profile(
    rollouts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["id"]): row for row in rows}
    vectors: list[list[int]] = []
    keyed: dict[str, list[int]] = {}
    for item in rollouts:
        instance_id = str(item["instance_id"])
        vector = cumulative_conditions(str(item["completion"]), by_id[instance_id])
        vectors.append(vector)
        key = f"{instance_id}::{int(item['rollout_index'])}"
        keyed[key] = vector
    counts = [sum(vector[index] for vector in vectors) for index in range(5)]
    total = len(vectors)
    return {
        "rollouts": total,
        "counts": counts,
        "rates": [count / total for count in counts],
        "condition_names": [
            "parse",
            "parse_and_finish_all",
            "parse_finish_and_prefix_all",
            "parse_finish_prefix_and_residual_all",
            "official_full_pass",
        ],
        "distinct_rungs": 4,
        "keyed_vectors": keyed,
    }


def bottleneck_index(profile: dict[str, Any], delta: float) -> int:
    # One-based index. c4 and c5 coincide, so only the four distinct rungs
    # participate in bottleneck discovery.
    for index, rate in enumerate(profile["rates"][:4], start=1):
        if float(rate) < 1.0 - delta:
            return index
    return 4
