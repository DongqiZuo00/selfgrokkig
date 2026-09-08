from __future__ import annotations

import argparse
import time

from huggingface_hub import snapshot_download

from common import MODEL_REPO_ID, MODEL_ROOT, EXP_ROOT, atomic_json, ensure_dirs


REPO_ID = MODEL_REPO_ID
if "qwen" in REPO_ID.lower():
    raise RuntimeError("Qwen downloads are prohibited")


def model_ready() -> bool:
    return (
        (MODEL_ROOT / "config.json").exists()
        and (MODEL_ROOT / "tokenizer_config.json").exists()
        and (MODEL_ROOT / "chat_template.jinja").exists()
        and bool(list(MODEL_ROOT.glob("*.safetensors")))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    started = time.time()
    if not model_ready():
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(MODEL_ROOT),
            local_files_only=args.local_files_only,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "consolidated.safetensors"],
        )
    if not model_ready():
        raise RuntimeError(f"incomplete model snapshot: {MODEL_ROOT}")
    atomic_json(
        EXP_ROOT / "manifests" / "mistral_model_snapshot.json",
        {
            "repo_id": REPO_ID,
            "local_path": str(MODEL_ROOT),
            "ready": True,
            "safetensor_files": sorted(path.name for path in MODEL_ROOT.glob("*.safetensors")),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"model ready: {MODEL_ROOT}", flush=True)


if __name__ == "__main__":
    main()
