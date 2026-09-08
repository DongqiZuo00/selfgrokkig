from __future__ import annotations

from recipe_train_branch import bridge_verification_case_sets


def signatures(sets):
    return {tuple(int(case["index"]) for case in subset) for subset in sets}


def main() -> None:
    branch = {"seed": 42}
    three = [{"index": index} for index in range(3)]
    stage_three = {
        "verification_case_counts": [2, 2, 2, 2, 2, 2, 3, 3],
        "verification_subset_mode": "rotating_combinations",
    }
    bridge_row = {"ground_truth": three, "_coverage_bridge_verifier": True}
    target_row = {"ground_truth": three + [{"index": 3}]}
    assigned = bridge_verification_case_sets(
        branch, 2, stage_three, 15, [bridge_row, target_row], 8
    )
    assert {(0, 1), (0, 2), (1, 2)} <= signatures(assigned[0])
    assert all(len(subset) == 4 for subset in assigned[1])

    four = [{"index": index} for index in range(4)]
    stage_four = {
        "verification_case_counts": [3] * 8,
        "verification_subset_mode": "rotating_combinations",
    }
    leave_one_out = bridge_verification_case_sets(
        branch,
        6,
        stage_four,
        45,
        [{"ground_truth": four, "_coverage_bridge_verifier": True}],
        8,
    )
    assert signatures(leave_one_out[0]) == {
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    }
    print("v4.6 rotating subsets and full-target isolation: PASS")


if __name__ == "__main__":
    main()
