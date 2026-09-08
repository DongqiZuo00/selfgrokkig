"""Real native-token HF Solver backend for the bounded VERGE pilot.

No synthetic samples, reference programs or per-test fractions enter loss.
The previous release remains immutable. All paths are explicit selfgrok paths.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "verge_operational_20260908"
sys.path[:0] = [str(PRIOR / name) for name in ("runtime", "benchmark", "protocol", "stages")]
from benchmark import extract_program, evaluate_program
from generated_budget import binary_advantages, assert_fresh_optimizer
from solver_update import train_step
from live_smoke_prefill import prompt_text, trim_completion, optimizer_snapshot, same_state

WORK = Path("/blue/du.j/jinjiaguo/self grok")
MODEL = WORK / "models/Ministral-3-3B-Instruct-2512-BF16"
INITIAL_SOLVER = WORK / "experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_solver/resume_u0000"
INITIAL_CHALLENGER = WORK / "experiments/has_transfer_witness/checkpoints/verge_book_v2_initial_challenger/resume_u0000"
FORMAT_PREFIX = "```manufactoria\nSTART start:\n    NEXT "


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_cases(completion, cases, create_factory):
    """Every fixed test survives parse/runtime errors; only all-pass is reward."""
    if not cases or any(c.get("check_output") is not True or c.get("expected_accepted") is not True
                        or not isinstance(c.get("expected_output"), str) for c in cases):
        raise ValueError("Frozen exact-output test cases are required")
    program = extract_program(completion)
    try:
        factory = create_factory(program)
        error, parsed = None, True
    except Exception as exception:
        factory, error, parsed = None, str(exception), False
    outcomes = []
    for index, case in enumerate(cases):
        passed, reason, steps = False, error, 0
        if factory is not None:
            try:
                actual = factory.process_robot(case["input"])
                passed = bool(actual.finished and actual.final_tape == case["expected_output"])
                reason = None if passed else actual.rejection_reason or "exact_output_mismatch"
                steps = len(actual.path)
            except Exception as exception:
                reason = str(exception)
        outcomes.append({"test_id": case.get("test_id", str(index)), "input": case["input"],
                         "pass": int(passed), "reason": reason, "steps": steps})
    return {"reward": int(all(r["pass"] for r in outcomes)), "parse_valid": parsed,
            "program": program, "per_test": outcomes}


class Solver:
    def __init__(self, adapter=INITIAL_SOLVER, *, device=0, learning_rate=1e-5, grammar_enabled=False):
        import torch
        from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration
        from peft import PeftModel
        self.torch = torch
        self.device = f"cuda:{device}"
        self.adapter = Path(adapter).resolve()
        if not self.adapter.is_relative_to(WORK.resolve()):
            raise ValueError("Adapter must remain in the actual selfgrok workspace")
        cfg = json.loads((self.adapter / "adapter_config.json").read_text())
        if Path(cfg["base_model_name_or_path"]).resolve() != MODEL.resolve() or cfg["r"] != 16:
            raise ValueError("Unexpected backbone/adapter; never infer a different project model")
        if AutoConfig.from_pretrained(str(MODEL), local_files_only=True).model_type != "mistral3":
            raise ValueError("Unexpected model type")
        if not torch.cuda.is_available() or torch.cuda.device_count() <= device:
            raise RuntimeError("An allocated GPU is required")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                fix_mistral_regex=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = Mistral3ForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True,
                   dtype=torch.bfloat16, device_map=self.device, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, str(self.adapter), is_trainable=True, local_files_only=True)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if any("lora_" not in name for name, p in self.model.named_parameters() if p.requires_grad):
            raise ValueError("Only Solver LoRA may train")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate, betas=(.9, .95), weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        assert_fresh_optimizer(self.optimizer, beta=0, weight_decay=0)
        self.optimizer_steps = 0
        self.generated_tokens = 0
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos or [self.tokenizer.eos_token_id])
        sys.path.insert(0, str(WORK / "vendor/rl-grok-recipe"))
        from manufactoria.verifier.manufactoria_parser import create_robot_factory
        self.create_factory = create_robot_factory
        self.grammar = None
        if grammar_enabled:
            sys.path.insert(0, str(ROOT / "decoding"))
            from hf_grammar_bridge import GrammarService
            self.grammar = GrammarService(WORK / "envs/vllm/bin/python", MODEL,
                                          log_name=f"grammar_{__import__('os').getpid()}.stderr.log")
            if set(self.grammar.metadata["eos_token_ids"]) != self.eos_ids:
                raise ValueError("Grammar and model EOS IDs differ")

    def generate(self, row, n, cap, seed, *, output=None, target=False, prefix=FORMAT_PREFIX):
        torch = self.torch
        if not 1 <= n <= 32 or not 1 <= cap <= 2048:
            raise ValueError("Bounded n/cap required")
        if row.get("hint") is not None and target:
            raise ValueError("Target evaluation/training cannot carry a hint")
        text = prompt_text(self.tokenizer, row["messages"])
        if not text.endswith(FORMAT_PREFIX):
            raise AssertionError("Audited format prefix disappeared")
        text = text[:-len(FORMAT_PREFIX)] + prefix
        batch = self.tokenizer([text] * n, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        self.model.eval()
        torch.manual_seed(seed)
        started = time.time()
        sampling = dict(do_sample=True, temperature=1., top_p=1., top_k=0)
        if self.grammar is not None:
            if prefix != FORMAT_PREFIX:
                raise ValueError("DSL grammar requires its exact frozen prompt prefix")
            from hf_grammar_bridge import HFGrammarLogitsProcessor, native_sampling_kwargs
            sampling = native_sampling_kwargs()
            sampling["logits_processor"] = [HFGrammarLogitsProcessor(self.grammar)]
        with torch.no_grad():
            generated = self.model.generate(**batch, **sampling,
                        max_new_tokens=cap, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=sorted(self.eos_ids))
        prompt_ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()
        native = [trim_completion(seq[batch["input_ids"].shape[1]:].tolist(), self.eos_ids) for seq in generated]
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in native]
        raw = {"mode": "real_model_generation", "adapter": str(self.adapter), "instance_id": row["id"],
               "seed": seed, "n": n, "max_new_tokens": cap, "prompt": text, "prompt_token_ids": prompt_ids,
               "completion_token_ids": native, "native_suffixes": texts, "format_prefix": prefix,
               "verifier_completions": [prefix + text for text in texts], "sampling_temperature": 1.,
               "top_p": 1., "top_k": 0, "generated_tokens": sum(map(len, native)),
               "sampling_seconds": time.time() - started, "optimizer_steps_before": self.optimizer_steps}
        raw["decoding_policy"] = "dsl_grammar_v1" if self.grammar is not None else "unconstrained_native"
        raw["grammar_metadata"] = self.grammar.metadata if self.grammar is not None else None
        self.generated_tokens += raw["generated_tokens"]
        if output is not None:
            write_json(output, raw)  # Save sampled work before verifier execution.
        verifier = evaluate_program if target else verify_cases
        raw["verification"] = [verifier(text, row["ground_truth"], self.create_factory) for text in raw["verifier_completions"]]
        raw["rewards"] = [value["reward"] for value in raw["verification"]]
        raw["binary_successes"] = sum(raw["rewards"])
        raw["parse_valid"] = sum(value["parse_valid"] for value in raw["verification"])
        raw["mixed_group"] = n == 8 and len(set(raw["rewards"])) == 2
        if output is not None:
            write_json(output, raw)
        return raw

    def update(self, raw):
        torch = self.torch
        advantages = binary_advantages(raw["rewards"])
        items = [{"prompt_token_ids": raw["prompt_token_ids"], "completion_token_ids": ids}
                 for ids in raw["completion_token_ids"]]
        masks = [[1] * len(item["completion_token_ids"]) for item in items]
        before = [p.detach().clone() for p in self.parameters]
        old_scheduler = copy.deepcopy(self.scheduler.state_dict())
        old_optimizer = optimizer_snapshot(self.optimizer, torch) if not any(advantages) else None
        self.model.config.use_cache = False
        if self.grammar is None:
            if raw.get("decoding_policy", "unconstrained_native") != "unconstrained_native":
                raise ValueError("Guided samples require the matching masked policy loss")
            metric = train_step(self.model, None, self.optimizer, self.scheduler, items, advantages, masks,
                                {"kl_beta": 0}, self.optimizer_steps)
        else:
            if raw.get("decoding_policy") != "dsl_grammar_v1" or raw.get("grammar_metadata") != self.grammar.metadata:
                raise ValueError("Training must use this generation grammar and native policy")
            from hf_grammar_bridge import masked_suffix_loss
            used = sum(map(len, raw["completion_token_ids"]))
            loss_sum = 0.
            if any(advantages):
                self.optimizer.zero_grad(set_to_none=True)
                self.model.train()
                for ids, advantage in zip(raw["completion_token_ids"], advantages):
                    packed, _ = self.grammar.replay(ids)
                    loss, logp = masked_suffix_loss(self.model, raw["prompt_token_ids"], ids, advantage, packed)
                    (loss / used).backward()
                    loss_sum += float(loss.detach())
                    del loss, logp, packed
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite Solver gradient; checkpoint not committed")
                self.optimizer.step()
                self.scheduler.step()
            metric = {"tokens_used": used, "nonzero_advantage_tokens": used if any(advantages) else 0,
                      "optimizer_step": bool(any(advantages)), "loss": loss_sum / used,
                      "loss_policy": "same_dsl_grammar_masked_native_suffix"}
        changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, self.parameters))
        if not any(advantages):
            if changed or self.scheduler.state_dict() != old_scheduler or not same_state(old_optimizer, self.optimizer.state_dict(), torch):
                raise AssertionError("Constant reward group changed model or optimizer")
        elif not changed or not metric["optimizer_step"]:
            raise AssertionError("Mixed reward failed to update actual Solver LoRA")
        self.optimizer_steps += int(metric["optimizer_step"])
        return {**metric, "parameters_changed": changed, "optimizer_steps": self.optimizer_steps,
                "beta": 0, "weight_decay": 0, "binary_reward_only": True}

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError("Checkpoint output must be new")
        self.model.save_pretrained(path)
        write_json(path / "pilot_state.json", {"source_adapter": str(self.adapter),
                    "generated_tokens": self.generated_tokens, "optimizer_steps": self.optimizer_steps})
        return str(path)

    def close(self):
        if self.grammar is not None:
            self.grammar.close()
