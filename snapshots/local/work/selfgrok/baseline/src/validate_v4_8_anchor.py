from __future__ import annotations

from recipe_train_branch import bridge_verification_case_sets


def signatures(sets):
    return {tuple(int(case["index"]) for case in subset) for subset in sets}


def main() -> None:
    branch = {"seed": 42}
    cases = [{"index": index} for index in range(3)]
    bridge_row = {"ground_truth": cases, "_coverage_bridge_verifier": True}
    target_row = {"ground_truth": cases}
    base = {
        "verification_subset_mode": "required_anchor_combinations",
        "verification_required_case_index": 2,
    }
    single = bridge_verification_case_sets(
        branch,
        0,
        {**base, "verification_case_counts": [1] * 8},
        1,
        [bridge_row, target_row],
        8,
    )
    assert signatures(single[0]) == {(2,)}
    assert signatures(single[1]) == {(0, 1, 2)}

    pairs = bridge_verification_case_sets(
        branch,
        1,
        {**base, "verification_case_counts": [2] * 8},
        9,
        [bridge_row],
        8,
    )
    assert signatures(pairs[0]) == {(0, 2), (1, 2)}
    assert all(2 in signature for signature in signatures(pairs[0]))

    mixed = bridge_verification_case_sets(
        branch,
        2,
        {**base, "verification_case_counts": [2, 2, 2, 2, 3, 3, 3, 3]},
        17,
        [bridge_row],
        8,
    )
    assert all(2 in signature for signature in signatures(mixed[0]))
    assert (0, 1, 2) in signatures(mixed[0])
    print("v4.8 required-case anchor and full-target isolation: PASS")


if __name__ == "__main__":
    main()
