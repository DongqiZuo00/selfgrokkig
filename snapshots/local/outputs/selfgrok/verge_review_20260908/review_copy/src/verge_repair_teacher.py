"""Schema-normalized on-policy teacher loss with native sampler likelihood check."""
import json
import numpy as np
import torch

from common import EXP_ROOT, read_json, atomic_json
from train_branch import build_model
from verge_round_core import ROOT, VERSION
from verge_round_runtime import optimizer_for, commit, free_models
from verge_repair_training import token_loss, disable_training_dropout
from verge_repair_protocol import assert_challenger_likelihood_is_matched


def update_teacher(frozen, decision):
    output = ROOT / "challenger_update.json"
    if output.exists():
        return read_json(output)
    cfg = frozen["config"]
    assert_challenger_likelihood_is_matched(cfg["challenger_sampling"], cfg["challenger_loss"])
    advantages = decision["challenger_advantages"]
    updated = decision["challenger_update_allowed"] and any(abs(a) > 1e-12 for a in advantages)
    destination = EXP_ROOT / "checkpoints" / (VERSION + "_challenger") / "resume_u0001"
    if updated and (destination / "verge_committed.json").exists():
        saved = torch.load(destination / "state.pt", map_location="cpu", weights_only=False)
        result = saved["runtime_state"]["teacher_result"]
        atomic_json(output, result)
        return result
    tokenizer, model = build_model(42, EXP_ROOT / frozen["teacher_checkpoint"])
    optimizer, scheduler = optimizer_for(model, cfg["challenger_learning_rate"])
    prior_steps = 0
    if cfg.get("book_suite"):
        from verge_book_runtime import inherit_optimizer
        prior_steps = inherit_optimizer(optimizer, scheduler, EXP_ROOT / frozen["teacher_checkpoint"])
    disable_training_dropout(model)
    optimizer.zero_grad(set_to_none=True)
    masks = np.load(ROOT / "challenger_masks.npz", allow_pickle=False)
    checks, losses = [], []
    candidates = frozen["generation"]["candidates"]
    prompt = frozen["generation"]["prompt_token_ids"]
    for candidate, advantage in zip(candidates, advantages):
        ids = candidate["completion_token_ids"]
        if masks[f"tokens_{candidate['index']}"].tolist() != ids:
            raise RuntimeError("Grammar masks belong to different sampled actions")
        packed = masks[f"candidate_{candidate['index']}"]
        # Validate likelihoods even if all real rewards tie. No artificial reward
        # is substituted, and zero-variance groups leave the teacher unchanged.
        with torch.set_grad_enabled(updated):
            loss, logp = token_loss(model, prompt, ids, advantage, grammar=packed)
            observed = candidate["sampler_logprobs"]
            if observed is None or len(observed) != len(ids) or any(v is None for v in observed):
                raise RuntimeError("Missing processed native log probabilities; cannot verify masked learner")
            errors = np.abs(logp.detach().float().cpu().numpy() - np.asarray(observed))
            check = {"candidate": candidate["index"], "tokens": len(ids),
                     "mean_absolute_error": float(errors.mean()), "max_absolute_error": float(errors.max())}
            if not np.isfinite(errors).all() or errors.mean() > .08 or errors.max() > .75:
                atomic_json(ROOT / "likelihood_failure.json", check)
                raise RuntimeError("Native sampler/learner likelihood mismatch exceeds BF16 tolerance")
            checks.append(check)
            if updated:
                normalized = loss / len(ids) / len(candidates)
                normalized.backward()
                losses.append(float(normalized.detach()))
        del loss, logp
    masks.close()
    result = {"updated": updated, "optimizer_steps": int(updated),
        "checkpoint": str(destination.relative_to(EXP_ROOT)) if updated else frozen["teacher_checkpoint"],
        "whole_proposal_advantages": advantages, "rewards": decision["challenger_rewards"],
        "losses": losses, "likelihood_checks": checks,
        "algorithm": "one schema-normalized, token-mean policy-gradient step with whole-proposal credit",
        "sampling_distribution": cfg["challenger_sampling"], "loss_distribution": cfg["challenger_loss"],
        "zero_variance_convention": "zero advantages, retain Challenger", "teacher_kl_beta": 0.,
        "invalid_credit_policy": cfg["invalid_credit_policy"], "acceptance_only": cfg["acceptance_only"]}
    if cfg.get("book_suite"):
        result.update(cumulative_optimizer_steps=prior_steps + int(updated),
                      inherited_optimizer=True, teacher_reward_mode=cfg["teacher_reward"])
    if updated:
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite Challenger gradient")
        result["gradient_norm_before_clip"] = float(norm)
        optimizer.step()
        scheduler.step()
        commit(model, optimizer, scheduler, VERSION + "_challenger", 1, {"teacher_result": result})
    atomic_json(output, result)
    del model, tokenizer, optimizer, scheduler
    free_models()
    return result
