from __future__ import annotations

from datasets import load_dataset

from common import DATA_ROOT, EXP_ROOT, atomic_json, load_config, read_json


def main() -> None:
    marker = EXP_ROOT / "manifests" / "official_test_unsealed.json"
    if marker.exists():
        print("official test already unsealed", flush=True)
        return
    final_budget_path = EXP_ROOT / "manifests" / "final_common_budget.json"
    if not final_budget_path.exists():
        raise RuntimeError("candidate set, configuration, and final common budget are not frozen")
    frozen = read_json(EXP_ROOT / "manifests" / "frozen_manifest.json")
    budget = int(read_json(final_budget_path)["budget"])
    for branch in frozen["branches"]:
        summary = (
            EXP_ROOT
            / "raw_results"
            / "branches"
            / branch["branch_id"]
            / "confirmation"
            / f"budget_{budget}_summary.json"
        )
        if not summary.exists():
            raise RuntimeError(f"confirmation incomplete: {branch['branch_id']}")
    config = load_config()
    dataset = load_dataset(
        "manufactoria/prepend_sequence_test",
        cache_dir=str(DATA_ROOT.parent / "caches" / "huggingface" / "datasets"),
    )["train"]
    if len(dataset) != int(config["data"]["official_test"]):
        raise RuntimeError(f"expected 108 official test rows, received {len(dataset)}")
    path = DATA_ROOT / "official_sealed" / "prepend_sequence_test"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    atomic_json(
        marker,
        {
            "repository": "manufactoria/prepend_sequence_test",
            "rows": len(dataset),
            "dataset_fingerprint": dataset._fingerprint,
            "unsealed_after_common_budget": budget,
            "path": str(path),
        },
    )
    print(f"unsealed {len(dataset)} official PREPEND test rows after budget {budget}", flush=True)


if __name__ == "__main__":
    main()
