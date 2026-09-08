"""Failure-driven format repair after real GPU smoke 41369486 (retained).

One train prompt x eight native suffixes x <=2048 tokens, 17-minute GPU cap.
The verified production PROGRAM_PREFIX is prompt context only. The verifier
receives prefix+suffix, while the binary-reward loss uses suffix tokens only.
This is a pipeline smoke, not a VERGE round, regime test, or effectiveness claim.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "benchmark"), str(HERE / "protocol")]
from benchmark import evaluate_program
from generated_budget import binary_advantages, assert_fresh_optimizer, JsonlJournal

WORK = Path("/blue/du.j/jinjiaguo/self grok")
MODEL = WORK / "models/Ministral-3-3B-Instruct-2512-BF16"
ADAPTER = WORK / "experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000"
FORMAT_INTERFACE_SOURCE = WORK / "experiments/has_transfer_witness/src/verge_repair_protocol.py"
PROGRAM_PREFIX = "```manufactoria\nSTART start:\n    NEXT "
GROUP_COUNT = 1
GROUP_SIZE = 8
MAX_NEW_TOKENS = 2048
MAX_GENERATED_TOKENS = GROUP_COUNT * GROUP_SIZE * MAX_NEW_TOKENS
GPU_MINUTES_CEILING = 17
FAILURE_SOURCE_JOB_ID = "41369486"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trim_completion(values, eos_ids):
    """Include the generated EOS; remove only post-EOS batch padding."""
    result = []
    for value in values:
        result.append(int(value))
        if value in eos_ids:
            break
    if not result:
        raise ValueError("Empty generation cannot be silently excluded")
    return result


def prompt_text(tokenizer, messages):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PROGRAM_PREFIX
    text = tokenizer.bos_token or "<s>"
    for message in messages:
        if message["role"] != "user":
            raise ValueError("This bounded smoke expects only a target user prompt")
        text += "[INST]" + message["content"] + "[/INST]"
    return text + PROGRAM_PREFIX


def full_program(native_suffix):
    """Format prefill belongs to context, never to generated-token budget/loss."""
    return PROGRAM_PREFIX + native_suffix


def verify_format_source(path):
    """Read one literal from its verified production source without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "PROGRAM_PREFIX" for target in node.targets)]
    if len(definitions) != 1 or ast.literal_eval(definitions[0]) != PROGRAM_PREFIX:
        raise ValueError("Production format prefix no longer matches the audited interface")


def optimizer_snapshot(optimizer, torch):
    """Snapshot actual moments/counters; constant groups must preserve all of them."""
    def clone(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(clone(item) for item in value)
        return copy.deepcopy(value)
    return clone(optimizer.state_dict())


def same_state(left, right, torch):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_state(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(same_state(a, b, torch) for a, b in zip(left, right))
    return left == right


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    output = args.output.resolve()
    if not HERE.is_relative_to(WORK.resolve()):
        raise ValueError("Run the uploaded release inside the authorized selfgrok workspace")
    if not output.is_relative_to((WORK / "experiments/has_transfer_witness/outputs/verge_operational_20260908").resolve()):
        raise ValueError("Outputs must stay within the independent selfgrok release")
    if output.exists():
        raise FileExistsError("Use a new smoke directory; preserve previous logs/checkpoint")
    for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
        location = os.environ.get(name)
        if not location or not Path(location).resolve().is_relative_to(WORK.resolve()):
            raise ValueError(f"{name} must be explicitly confined to selfgrok")
    output.mkdir(parents=True)
    journal = JsonlJournal(output / "events.jsonl", output.name)
    import torch
    from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration
    from peft import PeftModel
    from solver_update import train_step
    sys.path.insert(0, str(WORK / "vendor/rl-grok-recipe"))
    from manufactoria.verifier.manufactoria_parser import create_robot_factory
    started = time.time()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke requires exactly one allocated GPU")
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    if config.model_type != "mistral3":
        raise ValueError("Actual backbone differs from audited Ministral3")
    verify_format_source(FORMAT_INTERFACE_SOURCE)
    manifests = {str(p): digest(p) for p in [MODEL / "config.json", MODEL / "tokenizer_config.json",
                 ADAPTER / "adapter_config.json", ADAPTER / "adapter_model.safetensors",
                 ADAPTER / "state.pt", Path(__file__), FORMAT_INTERFACE_SOURCE, HERE / "runtime/solver_update.py",
                 HERE / "benchmark/benchmark.py", HERE / "benchmark/generated/manifest.json",
                 HERE / "benchmark/generated/target_train.jsonl"]}
    adapter_config = json.loads((ADAPTER / "adapter_config.json").read_text())
    if adapter_config["r"] != 16 or adapter_config["lora_alpha"] != 32:
        raise ValueError("Actual Solver adapter differs from audited initial adapter")
    if Path(adapter_config["base_model_name_or_path"]).resolve() != MODEL.resolve():
        raise ValueError("Actual Solver adapter names a different backbone")
    initial_state = torch.load(ADAPTER / "state.pt", map_location="cpu", weights_only=True)
    runtime_state = initial_state.get("runtime_state", {})
    if initial_state.get("update") != 0 or runtime_state.get("role") != "solver" or runtime_state.get("trained_tokens") != 0:
        raise ValueError("Source checkpoint is not the audited untrained Solver initialization")
    manifest = json.loads((HERE / "benchmark/generated/manifest.json").read_text())
    if manifest["split_rows"]["train"]["sha256"] != manifests[str(HERE / "benchmark/generated/target_train.jsonl")]:
        raise ValueError("Train dataset does not match its frozen benchmark manifest")
    journal.append("preflight", {"model": str(MODEL), "adapter": str(ADAPTER), "sha256": manifests,
                   "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "groups": GROUP_COUNT, "group_size": GROUP_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
                   "max_generated_tokens": MAX_GENERATED_TOKENS, "budget_b200_hours_ceiling": GPU_MINUTES_CEILING / 60,
                   "binary_reward_only": True, "beta": 0, "weight_decay": 0,
                   "heldout_or_test_model_results_read": False, "mode": "failure_driven_format_prefill_pipeline_smoke",
                   "initial_runtime_state": runtime_state, "format_prefix": PROGRAM_PREFIX,
                   "format_interface_source": str(FORMAT_INTERFACE_SOURCE),
                   "failure_source_job_id": FAILURE_SOURCE_JOB_ID,
                   "repair_reason": "real prior smoke exposed missing production format prefill and verifier extraction",
                   "prefix_in_prompt_only": True, "loss_uses_generated_suffix_only": True,
                   "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "stop_strings": None}})
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                             fix_mistral_regex=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no usable explicit padding token")
    torch.manual_seed(20260908)
    base = Mistral3ForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True,
                    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=True, local_files_only=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    named_trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_trainable or any("lora_" not in name for name, _ in named_trainable):
        raise ValueError("Only the independently loaded Solver LoRA may be trainable")
    trainable = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, betas=(0.9, 0.95), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    assert_fresh_optimizer(optimizer, beta=0, weight_decay=0)
    rows = [json.loads(line) for line in (HERE / "benchmark/generated/target_train.jsonl").read_text().splitlines()][:GROUP_COUNT]
    if len(rows) != GROUP_COUNT:
        raise ValueError("The bounded prefill smoke requires one actual train instance")
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [tokenizer.eos_token_id])
    if not eos_ids or any(type(value) is not int or value < 0 for value in eos_ids):
        raise ValueError("A valid explicit EOS set is required for generation accounting")
    journal.append("loaded", {"trainable_parameters": sum(p.numel() for p in trainable),
                   "trainable_parameter_names": [name for name, _ in named_trainable],
                   "eos_token_ids": sorted(eos_ids), "pad_token_id": tokenizer.pad_token_id,
                   "optimizer_initial_state_entries": len(optimizer.state)})
    steps = total_tokens = total_successes = total_parse_valid = constant_groups = 0
    for group, row in enumerate(rows):
        if row.get("hint") is not None:
            raise ValueError("Target smoke must be unhinted")
        text = prompt_text(tokenizer, row["messages"])
        batch = tokenizer([text] * GROUP_SIZE, return_tensors="pt", add_special_tokens=False, padding=True).to("cuda")
        model.eval()
        torch.manual_seed(20260908 + group)
        journal.append("generation_attempt", {"group": group, "instance_id": row["id"],
                       "count": GROUP_SIZE, "max_new_tokens": MAX_NEW_TOKENS, "format_prefix": PROGRAM_PREFIX})
        with torch.no_grad():
            sampled = model.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                    max_new_tokens=MAX_NEW_TOKENS, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=sorted(eos_ids))
        prompt_ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()
        completions = [trim_completion(seq[batch["input_ids"].shape[1]:].tolist(), eos_ids) for seq in sampled]
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completions]
        verifier_completions = [full_program(suffix) for suffix in texts]
        total_tokens += sum(map(len, completions))
        journal.append("raw_generation", {"group": group, "instance_id": row["id"],
                       "prompt_token_ids": prompt_ids, "completion_token_ids": completions,
                       "completions": texts, "native_suffixes": texts,
                       "verifier_completions": verifier_completions, "format_prefix": PROGRAM_PREFIX,
                       "prefix_in_prompt_only": True, "loss_uses_generated_suffix_only": True,
                       "total_generated_tokens": total_tokens,
                       "generated_eos_included": True, "post_eos_padding_excluded": True})
        if len(completions) != GROUP_SIZE or any(len(ids) > MAX_NEW_TOKENS for ids in completions) or total_tokens > MAX_GENERATED_TOKENS:
            raise AssertionError("Generation violated the bounded smoke budget")
        verified = [evaluate_program(program, row["ground_truth"], create_robot_factory) for program in verifier_completions]
        rewards = [v["reward"] for v in verified]
        advantages = binary_advantages(rewards)
        items = [{"prompt_token_ids": prompt_ids, "completion_token_ids": ids} for ids in completions]
        masks = [[1] * len(ids) for ids in completions]
        before = [p.detach().clone() for p in trainable]
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        optimizer_before = optimizer_snapshot(optimizer, torch) if not any(advantages) else None
        journal.append("sampled_group", {"group": group, "instance_id": row["id"],
                       "prompt_token_ids": prompt_ids, "completion_token_ids": completions,
                       "completions": texts, "native_suffixes": texts,
                       "verifier_completions": verifier_completions, "format_prefix": PROGRAM_PREFIX,
                       "verification": verified, "rewards": rewards})
        model.config.use_cache = False
        metric = train_step(model, None, optimizer, scheduler, items, advantages, masks,
                            {"kl_beta": 0}, steps)
        moved = any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable))
        if not any(advantages):
            constant_groups += 1
            if moved or scheduler.state_dict() != scheduler_before or not same_state(optimizer_before, optimizer.state_dict(), torch):
                raise AssertionError("A constant reward group moved Solver, optimizer state or scheduler")
            if metric["optimizer_step"]:
                raise AssertionError("A constant reward group reported an optimizer step")
        elif not metric["optimizer_step"] or not moved:
            raise AssertionError("A mixed binary reward group failed to update Solver")
        steps += int(metric["optimizer_step"])
        total_successes += sum(rewards)
        total_parse_valid += sum(v["parse_valid"] for v in verified)
        journal.append("update", {"group": group, "metric": metric, "parameters_changed": moved,
                       "total_generated_tokens": total_tokens, "optimizer_steps": steps})
        print(json.dumps({"group": group, "binary_successes": sum(rewards), "parse_valid": sum(v["parse_valid"] for v in verified),
                          "generated_tokens": total_tokens, "optimizer_steps": steps}), flush=True)
        del before, optimizer_before, sampled, batch
    model.save_pretrained(output / "solver_adapter")
    summary = {"status": "complete", "mode": "failure_driven_format_prefill_pipeline_smoke", "groups": len(rows),
               "rollouts": len(rows) * GROUP_SIZE, "generated_tokens": total_tokens, "binary_successes": total_successes,
               "parse_valid": total_parse_valid, "max_generated_tokens": MAX_GENERATED_TOKENS,
               "max_new_tokens": MAX_NEW_TOKENS, "gpu_minutes_ceiling": GPU_MINUTES_CEILING,
               "failure_source_job_id": FAILURE_SOURCE_JOB_ID, "format_prefix": PROGRAM_PREFIX,
               "prefix_in_prompt_only": True, "loss_uses_generated_suffix_only": True,
               "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "stop_strings": None},
               "optimizer_steps": steps, "elapsed_seconds": time.time() - started,
               "constant_groups_exact_skip_checked": constant_groups,
               "mixed_binary_update_exercised": steps > 0,
               "script_section_gpu_hours": (time.time() - started) / 3600,
               "allocation_cost_source": "Slurm sacct Elapsed times allocated GPU count; script timing excludes allocation startup",
               "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
               "source_adapter_unchanged": all(digest(ADAPTER / name) == manifests[str(ADAPTER / name)]
                                               for name in ("adapter_config.json", "adapter_model.safetensors", "state.pt")),
               "benchmark_regime_established": False, "formal_verge_round": False,
               "original_sources_changed": False}
    if not summary["source_adapter_unchanged"]:
        raise AssertionError("Source adapter changed during smoke")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    journal.append("complete", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
