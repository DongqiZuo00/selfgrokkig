from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from common import DATA_ROOT, EXP_ROOT, atomic_json, read_json
from prepare_data import generate_distribution


def validate_existing(path: Path, candidate_id: str, expected: int) -> bool:
    if not path.exists():
        return False
    dataset = load_from_disk(str(path))
    if len(dataset) != expected:
        raise RuntimeError(f"existing {path} has {len(dataset)} rows, expected {expected}")
    if set(map(str, dataset["candidate_id"])) != {candidate_id}:
        raise RuntimeError(f"existing {path} has incompatible candidate IDs")
    return True


def main() -> None:
    protocol = read_json(EXP_ROOT / "manifests" / "soft_phi_rollout_round1.json")
    count = 480
    records: list[dict[str, Any]] = []
    for proposal in protocol["challenger_round0"]["proposals"]:
        candidate_id = str(proposal["candidate_id"])
        path = DATA_ROOT / "soft_phi_round1" / candidate_id / "training"
        if not validate_existing(path, candidate_id, count):
            rows = generate_distribution(
                str(proposal["family"]),
                dict(proposal["configuration"]),
                count,
                314159,
                "soft_phi_round1_training",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).save_to_disk(str(path))
        dataset = load_from_disk(str(path))
        records.append(
            {
                "candidate_id": candidate_id,
                "family": proposal["family"],
                "rows": len(dataset),
                "path": str(path),
            }
        )
    root = EXP_ROOT / "checkpoints" / "recipe__root__basic__seed42" / "resume_u0100"
    if not (root / "adapter_config.json").exists():
        raise RuntimeError(f"missing common parent: {root}")
    atomic_json(
        EXP_ROOT / "manifests" / "soft_phi_round1_data_ready.json",
        {
            "protocol_version": protocol["protocol_version"],
            "candidate_training_data": records,
            "common_parent": str(root),
            "target_development_used": False,
            "confirmation_opened": False,
            "official_test_opened": False,
        },
    )
    print(f"prepared {len(records)} target-blind proposals", flush=True)


if __name__ == "__main__":
    main()
