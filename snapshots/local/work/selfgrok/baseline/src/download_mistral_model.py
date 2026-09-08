from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

from common import EXP_ROOT, MODEL_REPO_ID, MODEL_ROOT, atomic_json, ensure_dirs


def complete(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and (path / "tokenizer_config.json").exists()
        and (path / "chat_template.jinja").exists()
        # vLLM 0.21 + Transformers 4 needs the native Mistral checkpoint to
        # select the Mistral config parser.  The HF shards remain necessary
        # for the Transformers learner, so this is not redundant at runtime.
        and (path / "consolidated.safetensors").exists()
        and bool(list(path.glob("*.safetensors")))
    )


def main() -> None:
    ensure_dirs()
    if "qwen" in MODEL_REPO_ID.lower() or "qwen" in str(MODEL_ROOT).lower():
        raise RuntimeError("Qwen model downloads are prohibited")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if not complete(MODEL_ROOT):
        snapshot_download(
            repo_id=MODEL_REPO_ID,
            local_dir=str(MODEL_ROOT),
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.model",
                "*.safetensors",
                "*.py",
                "tokenizer*",
                "tekken*",
            ],
        )
    if not complete(MODEL_ROOT):
        raise RuntimeError(f"incomplete Mistral snapshot: {MODEL_ROOT}")
    record = {
        "repo_id": MODEL_REPO_ID,
        "local_path": str(MODEL_ROOT),
        "qwen_prohibited": True,
        "safetensor_files": sorted(path.name for path in MODEL_ROOT.glob("*.safetensors")),
    }
    atomic_json(EXP_ROOT / "manifests" / "mistral_model_snapshot.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
