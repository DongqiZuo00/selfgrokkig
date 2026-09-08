from __future__ import annotations

import random

from datasets import Dataset

from common import DATA_ROOT, EXP_ROOT, atomic_json, stable_int
from prepare_data import generate_distribution


FAMILIES = ("contains_substring", "contains_ordered", "contains_count")
LEVELS = (
    {
        "name": "r6_level1_atomic_two_color",
        "color_mode": "two_color",
        "sequence_lengths": [1, 2],
        "count_thresholds": [1, 3],
        "numerical_thresholds": [1, 3],
        "regex_max_pattern_length": [1, 1],
    },
    {
        "name": "r6_level2_composed_four_color",
        "color_mode": "four_color",
        "sequence_lengths": [2, 3],
        "count_thresholds": [2, 5],
        "numerical_thresholds": [2, 5],
        "regex_max_pattern_length": [1, 2],
    },
    {
        "name": "r6_level3_target_near_four_color",
        "color_mode": "four_color",
        "sequence_lengths": [3, 4],
        "count_thresholds": [3, 8],
        "numerical_thresholds": [3, 8],
        "regex_max_pattern_length": [2, 3],
    },
)


def main() -> None:
    root = DATA_ROOT / "self_grok_bridge_chain_round6"
    entries = []
    per_family = (534, 533, 533)
    for level_index, config in enumerate(LEVELS, start=1):
        rows = []
        family_counts = {}
        for family, count in zip(FAMILIES, per_family):
            generated = generate_distribution(
                family,
                dict(config),
                count,
                stable_int("self_grok_r6", level_index, family),
                f"r6_level_{level_index}_training",
            )
            rows.extend(generated)
            family_counts[family] = len(generated)
        random.Random(stable_int("self_grok_r6_shuffle", level_index)).shuffle(rows)
        path = root / f"level_{level_index}"
        Dataset.from_list(rows).save_to_disk(str(path))
        entries.append(
            {
                "level": level_index,
                "name": config["name"],
                "data_path": str(path.relative_to(EXP_ROOT)),
                "rows": len(rows),
                "families": family_counts,
                "configuration": config,
            }
        )
    atomic_json(
        EXP_ROOT / "manifests" / "self_grok_bridge_chain_round6_data.json",
        {
            "status": "prepared",
            "purpose": "training-only target-aware component bridge; not an audit",
            "levels": entries,
        },
    )
    print(f"prepared {len(entries)} bridge levels", flush=True)


if __name__ == "__main__":
    main()
