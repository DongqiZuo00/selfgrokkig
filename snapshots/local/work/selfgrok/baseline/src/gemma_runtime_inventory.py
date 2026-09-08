"""Read installed package metadata and architecture source presence; no GPU/model imports."""
import importlib.metadata as metadata
import argparse
import json
import time
from pathlib import Path


def package(name, relative_sources):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"installed": False}
    result = {"installed": True, "version": dist.version, "sources": {}}
    for relative in relative_sources:
        path = Path(dist.locate_file(relative))
        if not path.is_file():
            result["sources"][relative] = {"exists": False}
            continue
        code = path.read_text(encoding="utf-8")
        result["sources"][relative] = {"exists": True,
            "mentions_unified_architecture": "Gemma4Unified" in code or "gemma4_unified" in code}
    return result


def inspect():
    return {"scope": "Installed architecture source inventory only; no claim of runtime training compatibility",
        "packages": {
            "transformers": package("transformers", [
                "transformers/models/gemma4_unified/configuration_gemma4_unified.py",
                "transformers/models/gemma4_unified/modeling_gemma4_unified.py",
                "transformers/models/gemma4_unified/processing_gemma4_unified.py",
                "transformers/models/auto/configuration_auto.py"]),
            "vllm": package("vllm", ["vllm/model_executor/models/registry.py",
                "vllm/model_executor/models/gemma4_unified.py"]),
            "torch": package("torch", []), "peft": package("peft", [])},
        "model_weights_downloaded": False, "model_loaded": False,
        "model_samples": 0, "gpu_allocation": 0, "existing_environment_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-label", choices=("training", "sampling"))
    args = parser.parse_args()
    result = inspect()
    result["observed_at_unix"] = time.time()
    if args.save_label:
        from verge_book_controller import EXP, write
        write(EXP / "outputs" / f"gemma_runtime_inventory_{args.save_label}.json", result)
    print(json.dumps(result, indent=2))
