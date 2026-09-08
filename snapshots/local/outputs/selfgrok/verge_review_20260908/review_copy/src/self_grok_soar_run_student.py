from __future__ import annotations

import argparse
import subprocess
import sys

from common import EXP_ROOT, read_json
from self_grok_soar_loop import generation_path
from vllm_runtime import VLLMServerPool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    generation = read_json(generation_path(args.outer))
    branch_id = str(generation["branch_ids"][args.index])
    with VLLMServerPool(gpus=(0,), base_port=args.port) as urls:
        subprocess.run(
            [
                sys.executable,
                str(EXP_ROOT / "src" / "recipe_train_branch.py"),
                "--branch-id",
                branch_id,
                "--server-url",
                urls[0],
                "--gpu",
                "0",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
