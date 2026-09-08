from __future__ import annotations

import argparse
import time

from common import EXP_ROOT, atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--branches", required=True)
    parser.add_argument("--analyze", required=True)
    parser.add_argument("--recovery-reason")
    args = parser.parse_args()
    path = EXP_ROOT / "manifests" / "recipe_submission.json"
    previous = read_json(path) if path.exists() else None
    atomic_json(
        path,
        {
            "submitted_at_unix": time.time(),
            "recovery_reason": args.recovery_reason,
            "previous_submission": previous,
            "prepare_job": args.prepare,
            "root_job": args.root,
            "screen_job": args.screen,
            "branch_array_job": args.branches,
            "analysis_job": args.analyze,
            "resource_ceiling": {
                "gpus": "2 x B200",
                "cpu_memory_gb": 64,
                "array_concurrency": 2,
            },
        },
    )


if __name__ == "__main__":
    main()
