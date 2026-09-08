"""Read-only CPU input/import preflight for the queued live_smoke.py.

No backbone weights are loaded. No task, model, adapter, manifest or queued
script is modified. The sole optional output is this independent report.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parents[1]
WORK = Path("/blue/du.j/jinjiaguo/self grok")
MODEL = WORK / "models/Ministral-3-3B-Instruct-2512-BF16"
ADAPTER = WORK / "experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires explicit CUDA_VISIBLE_DEVICES=''")
    if not HERE.resolve().is_relative_to(WORK.resolve()):
        raise RuntimeError("Run this preflight only in the authorized selfgrok release")
    if args.output and args.output.resolve().parent != (HERE / "runtime").resolve():
        raise RuntimeError("Report output must be in the independent release runtime directory")
    report = {"mode": "read_only_cpu_preflight", "started_unix": time.time(),
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
              "backbone_weights_loaded": False, "gpu_used": False,
              "checkpoint_or_queued_script_modified": False,
              "heldout_or_test_model_results_read": False, "checks": {}}
    checks = report["checks"]
    try:
        import torch
        import transformers
        import peft
        from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration
        from peft import PeftModel
        report["versions"] = {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "peft": peft.__version__}
        report["imports"] = {"Mistral3ForConditionalGeneration": Mistral3ForConditionalGeneration.__name__,
                             "PeftModel": PeftModel.__name__}
        checks["required_model_and_adapter_classes_import"] = True
        checks["cuda_devices_hidden"] = torch.cuda.device_count() == 0
        live_path = HERE / "runtime/live_smoke.py"
        live_source = live_path.read_text()
        tree = ast.parse(live_source)
        report["live_smoke_sha256"] = digest(live_path)
        expected_guards = [
            'config.model_type != "mistral3"',
            'adapter_config["r"] != 16 or adapter_config["lora_alpha"] != 32',
            'initial_state.get("update") != 0',
            'runtime_state.get("role") != "solver"',
            'runtime_state.get("trained_tokens") != 0',
            'weights_only=True',
            'manifest["split_rows"]["train"]["sha256"]',
        ]
        checks["queued_source_contains_expected_guards"] = all(s in live_source for s in expected_guards)
        constants = {node.targets[0].id: ast.unparse(node.value)
                     for node in tree.body if isinstance(node, ast.Assign)
                     and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ("WORK", "MODEL", "ADAPTER")}
        report["queued_source_constants"] = constants
        checks["queued_source_uses_audited_paths"] = (
            constants.get("WORK") == ast.unparse(ast.parse(f"Path({str(WORK)!r})", mode="eval").body)
            and constants.get("MODEL") == "WORK / 'models/Ministral-3-3B-Instruct-2512-BF16'"
            and constants.get("ADAPTER") == "WORK / 'experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000'")
        paths = [MODEL / "config.json", MODEL / "tokenizer_config.json", ADAPTER / "adapter_config.json",
                 ADAPTER / "state.pt", HERE / "benchmark/generated/manifest.json",
                 HERE / "benchmark/generated/target_train.jsonl"]
        for path in paths:
            if not path.resolve().is_relative_to(WORK.resolve()):
                raise RuntimeError(f"Input escapes authorized workspace: {path}")
        report["inputs"] = {str(p): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in paths}
        report["adapter_weights_metadata_only"] = {"path": str(ADAPTER / "adapter_model.safetensors"),
                                                   "size_bytes": (ADAPTER / "adapter_model.safetensors").stat().st_size,
                                                   "weights_loaded": False}
        config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
        report["model_config"] = {"model_type": config.model_type,
                                  "architectures": config.architectures,
                                  "text_model_type": getattr(getattr(config, "text_config", None), "model_type", None)}
        checks["model_type_mistral3"] = config.model_type == "mistral3"
        adapter_config = json.loads((ADAPTER / "adapter_config.json").read_text())
        report["adapter_config"] = {k: adapter_config.get(k) for k in (
            "base_model_name_or_path", "r", "lora_alpha", "peft_type", "task_type", "inference_mode")}
        checks["adapter_rank_and_alpha"] = adapter_config["r"] == 16 and adapter_config["lora_alpha"] == 32
        checks["adapter_backbone_source"] = Path(adapter_config["base_model_name_or_path"]).resolve() == MODEL.resolve()
        if (ADAPTER / "state.pt").stat().st_size > 256 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large initial state; inspect without loading")
        initial_state = torch.load(ADAPTER / "state.pt", map_location="cpu", weights_only=True)
        runtime_state = initial_state.get("runtime_state", {})
        report["initial_solver_state"] = {"top_level_keys": sorted(initial_state),
                                           "update": initial_state.get("update"),
                                           "runtime_state": {k: runtime_state.get(k) for k in (
                                               "role", "trained_tokens", "source", "initial_checkpoint", "base_checkpoint")}}
        checks["initial_solver_update_zero"] = initial_state.get("update") == 0
        checks["initial_solver_role"] = runtime_state.get("role") == "solver"
        checks["initial_solver_trained_tokens_zero"] = runtime_state.get("trained_tokens") == 0
        manifest = json.loads((HERE / "benchmark/generated/manifest.json").read_text())
        train_path = HERE / "benchmark/generated/target_train.jsonl"
        checks["train_manifest_sha256_matches"] = manifest["split_rows"]["train"]["sha256"] == digest(train_path)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                                 fix_mistral_regex=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        checks["tokenizer_has_padding_token"] = tokenizer.pad_token_id is not None
        rows = [json.loads(line) for line in train_path.read_text().splitlines()][:4]
        checks["four_unhinted_train_rows"] = len(rows) == 4 and all(row.get("hint") is None for row in rows)
        prompt_lengths = []
        for row in rows:
            if tokenizer.chat_template:
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            else:
                text = tokenizer.bos_token or "<s>"
                for message in row["messages"]:
                    if message["role"] != "user":
                        raise RuntimeError("Unexpected prompt role in fallback")
                    text += "[INST]" + message["content"] + "[/INST]"
            batch = tokenizer([text] * 8, return_tensors="pt", add_special_tokens=False, padding=True)
            prompt_lengths.append(int(batch["attention_mask"][0].sum()))
        report["tokenizer"] = {"class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side,
                               "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                               "has_chat_template": bool(tokenizer.chat_template),
                               "four_prompt_token_lengths": prompt_lengths}
        checks["four_eight_sample_prompts_tokenize"] = len(prompt_lengths) == 4 and min(prompt_lengths) > 0
        checks["read_only_inputs_unchanged"] = all(digest(Path(p)) == info["sha256"] for p, info in report["inputs"].items())
        checks["queued_live_smoke_unchanged"] = digest(live_path) == report["live_smoke_sha256"]
        report["status"] = "passed" if all(checks.values()) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["elapsed_seconds"] = time.time() - report["started_unix"]
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
