from __future__ import annotations

import argparse
import os
import subprocess
import sys

from common import EXP_ROOT, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    version = os.environ.get(
        "SELF_GROK_DECISIVE_VERSION", "self_grok_decisive_mistral_v4_1"
    )
    manifest = read_json(EXP_ROOT / "manifests" / f"{version}_runtime_branches.json")
    branch_id = manifest["branch_ids"][args.index]
    subprocess.run(
        [
            sys.executable,
            str(EXP_ROOT / "src" / "recipe_run_branch.py"),
            "--branch-id",
            branch_id,
            "--port",
            str(args.port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
