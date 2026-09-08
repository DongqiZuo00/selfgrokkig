"""Typed proposals, compact provenance and executable finite-map curriculum data."""
import json
import subprocess

from common import EXP_ROOT, MODEL_ROOT, read_json, atomic_json, apply_chat_template, stable_int
from modeling import load_tokenizer
from vllm_runtime import VLLM_PYTHON, http_json
from verge_round_core import ROOT, VERSION, generator_config
from verge_repair_protocol import (proposal_request, proposal_schema, parse_typed_proposal,
                                    repair_proposal_messages, finite_task)
from verge_repair_sampling import require_native_choice


def generate(cfg, initial, client, teacher):
    path = ROOT / "proposals.json"
    if path.exists():
        result = read_json(path)
    else:
        client.load_lora(VERSION + "_challenger", teacher)
        tokenizer = load_tokenizer()
        if cfg.get("book_suite"):
            from verge_book_protocol import proposal_messages
            from verge_book_runtime import history
            context = proposal_messages(cfg, initial, history(cfg))
        else:
            old_root = EXP_ROOT / "raw_results/verge_mistral_round1"
            prior_decision = read_json(old_root / "decision.json")
            prior = {"source": "completed round 1, old output interface",
                     "evaluated_candidates": prior_decision["evaluated_candidates"],
                     "target_full_pass": "0/512 start, direct and only valid course",
                     "legal_course": ["starts_with", "contains_substring", "contains_ordered"],
                     "stage_initial_successes": [31, 0, 0], "stage_initial_draws": 64}
            local = {"source": "v5.5 human-designed finite-task warm start, not the fixed target",
                     "observed_full_pass_before": "1/64", "observed_full_pass_after": "4/64",
                     "replay_auxiliary_weight": 0.20,
                     "limitation": "Different task and interface; not target transfer. No prescribed task or program supplied."}
            context = repair_proposal_messages(cfg["target_descriptor"], initial, prior, local)
        context[-1]["content"] += "\nTraining contract: " + json.dumps({
            "tokens_per_branch": cfg["train_tokens_per_branch"], "curriculum_budget_fraction": .75,
            "target_prompt_fraction": .25, "target_only_tail_fraction": .25,
            "update_threshold_tokens": cfg["token_update_threshold"], "completion_cap": 2048,
            "binary_task_rewards": True, "replay_loss": 0., "legal_families": cfg["families"]})
        prompt_ids = tokenizer(apply_chat_template(tokenizer, context), add_special_tokens=False).input_ids
        if len(prompt_ids) + cfg["proposal_max_tokens"] > 12288:
            raise RuntimeError("Compact Challenger context still exceeds the context budget")
        request = proposal_request(client.model_name, prompt_ids, cfg["families"], stable_int(cfg.get("random_stream_id", VERSION), 42, "proposals"))
        request["logprobs"] = 1
        response_path = ROOT / "proposal_response.json"
        if response_path.exists():
            stored = read_json(response_path)
            if stored["request"] != request:
                raise RuntimeError("Proposal request changed during recovery")
            response = stored["response"]
        else:
            response = http_json(client.base_url + "/v1/completions", request, timeout=3600)
            atomic_json(response_path, {"request": request, "response": response})
        choices = sorted(response["choices"], key=lambda c: c["index"])
        if len(choices) != 3:
            raise RuntimeError("Expected exactly three autonomous proposals")
        proposals = []
        for index, choice in enumerate(choices, 1):
            native = require_native_choice(choice, prompt_ids)
            try:
                spec = parse_typed_proposal(choice["text"], cfg["families"])
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                spec, error = None, str(exc)
            proposals.append({"index": index, "text": choice["text"], "spec": spec,
                "valid": spec is not None, "validation_error": error, "completion_token_ids": native,
                "sampler_logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
                "finish_reason": choice.get("finish_reason")})
        result = {"messages": context, "prompt_token_ids": prompt_ids, "candidates": proposals,
                  "schema": proposal_schema(cfg["families"]), "one_generation_request": True,
                  "usage": response.get("usage"), "sampling_distribution": cfg["challenger_sampling"]}
        atomic_json(path, result)
    if not all(p["valid"] for p in result["candidates"]):
        atomic_json(ROOT / "interface_failure.json", {"reason": "Constrained proposal group still invalid",
                    "challenger_update": False, "resampled": False, "proposals": result})
        raise RuntimeError("Invalid constrained group; no transfer-preference update or GPU branches submitted")
    masks = ROOT / "challenger_masks.npz"
    if not masks.exists():
        subprocess.run([str(VLLM_PYTHON), str(EXP_ROOT / "src/verge_repair_grammar.py"),
                        str(path), str(masks), str(MODEL_ROOT)], check=True, cwd=EXP_ROOT)
    return result


def stage_rows(stage, count, seed, label, template):
    if stage["family"] == "finite_map":
        return [finite_task(stage, template, f"{label}_{i:04d}") for i in range(count)]
    from prepare_data import generate_distribution
    return generate_distribution(stage["family"], generator_config(stage), count, seed, label)
