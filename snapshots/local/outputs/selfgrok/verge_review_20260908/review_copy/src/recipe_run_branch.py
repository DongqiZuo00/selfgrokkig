from __future__ import annotations

import argparse
import subprocess
import sys

from common import EXP_ROOT
from vllm_runtime import VLLMServerPool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    with VLLMServerPool(gpus=(0,), base_port=args.port) as urls:
        subprocess.run(
            [
                sys.executable,
                str(EXP_ROOT / "src" / "recipe_train_branch.py"),
                "--branch-id",
                args.branch_id,
                "--server-url",
                urls[0],
                "--gpu",
                "0",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
