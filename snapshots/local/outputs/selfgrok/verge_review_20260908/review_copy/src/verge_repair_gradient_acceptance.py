"""Disposable GPU backward-path test. Synthetic test credit, NEVER research evidence."""
import torch
from common import EXP_ROOT, read_json, read_jsonl, atomic_json
from recipe_train_branch import build_recipe_model, build_reference_model
from verge_round_core import ROOT
from verge_round_runtime import optimizer_for, free_models
from verge_repair_training import train_step, token_loss, disable_training_dropout


def check(frozen):
    cfg = frozen["config"]
    if not cfg["acceptance_only"]:
        raise RuntimeError("Synthetic-credit backward test is forbidden in the scientific round")
    # Reuse recorded tokens; no fresh model sampling, teacher demonstration or verifier audit.
    records = read_jsonl(ROOT / "branches/0/train/update_0001.jsonl")[:8]
    tokenizer, model = build_recipe_model(42, EXP_ROOT / cfg["solver_start"])
    reference = build_reference_model(42, EXP_ROOT / cfg["solver_start"])
    optimizer, scheduler = optimizer_for(model, cfg["solver_learning_rate"])
    before = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    masks = [[1]*len(r["completion_token_ids"]) for r in records]
    # A one-action nonzero test coefficient guarantees exercising the loss path,
    # even when all REAL acceptance-rollout rewards are zero. Never saved to a branch.
    metric = train_step(model, reference, optimizer, scheduler, records, [1.] + [0.]*7,
                        masks, cfg, prior_steps=0)
    changed = sum(not torch.equal(before[n], p.detach().cpu())
                  for n,p in model.named_parameters() if p.requires_grad)
    if not metric["optimizer_step"] or not changed:
        raise RuntimeError("Disposable real-model backward test did not change trainable parameters")
    result = {"passed": True, "synthetic_test_credit": True, "scientific_result": False,
              "new_rollouts": 0, "adapter_saved": False, "branch_checkpoints_modified": False,
              "changed_trainable_tensors": changed, "loss": metric["loss"]}
    del before, model, reference, optimizer, scheduler, tokenizer
    free_models()
    # Exercise the real teacher's MASKED backward path separately, even if its
    # genuine gains all tie. Again discard the model, do not commit any adapter.
    import numpy as np
    from train_branch import build_model
    tokenizer, model = build_model(42, EXP_ROOT / frozen["teacher_checkpoint"])
    optimizer, scheduler = optimizer_for(model, cfg["challenger_learning_rate"])
    disable_training_dropout(model)
    candidate = frozen["generation"]["candidates"][0]
    packed = np.load(ROOT / "challenger_masks.npz", allow_pickle=False)
    loss, _ = token_loss(model, frozen["generation"]["prompt_token_ids"],
                        candidate["completion_token_ids"], 1., grammar=packed["candidate_1"])
    (loss / len(candidate["completion_token_ids"])).backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.:
        raise RuntimeError("Disposable masked teacher backward test failed")
    optimizer.step()
    result["teacher_masked_backward_passed"] = True
    result["teacher_test_gradient_norm"] = float(norm)
    packed.close()
    atomic_json(ROOT / "DISPOSABLE_GRADIENT_TEST.json", result)
    del model, optimizer, scheduler, tokenizer, loss
    free_models()
    return result
