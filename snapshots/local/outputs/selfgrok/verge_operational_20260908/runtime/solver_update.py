"""Preserved review repair plus strict binary-only operational entry points.

Source: verge_review_20260908/review_copy/src/verge_repair_training.py.
The original phase_boundary_masks, dropout, logp, token_loss and train_step
functions below are copied verbatim; they are regression targets, not a new
claim about the old full protocol. New binary_solver_train_step wraps this
real Torch update with beta=0, weight_decay=0 and complete binary groups.
No backbone, checkpoint, CUDA device or dataset is inferred here.
"""
import random


def phase_boundary_masks(lengths, remaining, seed):
    if len(lengths) != 8 or any(type(n) is not int or not 1 <= n <= 2048 for n in lengths):
        raise ValueError("Expected one complete eight-action reward group at the fixed 2048 cap")
    if remaining <= 0:
        raise ValueError("No phase quota remaining")
    total = sum(lengths)
    if total <= remaining:
        return [[1] * n for n in lengths]
    # Only a PHASE boundary may mask loss positions. Every sampled program remains
    # intact for verification and context. Uniform positions avoid response-order bias.
    kept = set(random.Random(seed).sample(range(total), remaining))
    masks, offset = [], 0
    for n in lengths:
        masks.append([int(offset + j in kept) for j in range(n)])
        offset += n
    return masks


def disable_training_dropout(model):
    import torch
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def chosen_masked_logp(logits, targets, packed=None, chunk=32):
    """Normalize over the same allowed grammar actions; bit 1 means allowed."""
    import torch
    values = []
    vocab = logits.shape[-1]
    for begin in range(0, len(targets), chunk):
        part = logits[begin:begin+chunk].float()
        target = targets[begin:begin+chunk]
        if packed is not None:
            bits = torch.as_tensor(packed[begin:begin+chunk].copy(), dtype=torch.int32, device=part.device)
            ids = torch.arange(vocab, device=part.device)
            allowed = ((bits[:, ids // 32].long() >> (ids % 32)) & 1).bool()
            if not bool(allowed.gather(1, target[:, None]).all()):
                raise RuntimeError("Observed action is forbidden by the reconstructed grammar mask")
            part = part.masked_fill(~allowed, -float("inf"))
        values.append(part.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(part, dim=-1))
    return torch.cat(values)


def token_loss(model, prompt, completion, advantage, loss_mask=None, reference=None, kl_beta=0., grammar=None):
    import torch
    if not prompt or not completion:
        raise ValueError("Empty action or prompt")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    start = len(prompt) - 1
    targets = ids[0, len(prompt):]
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = output.logits[0, start:-1]
    logp = chosen_masked_logp(logits, targets, grammar)
    terms = -float(advantage) * logp
    if kl_beta:
        if reference is None or grammar is not None:
            raise ValueError("Solver KL needs its immutable reference; teacher KL is disabled")
        with torch.no_grad():
            reference_output = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            ref = chosen_masked_logp(reference_output.logits[0, start:-1], targets)
            del reference_output
        delta = ref - logp
        terms = terms + kl_beta * (torch.exp(delta) - delta - 1.)
    mask = torch.ones_like(terms) if loss_mask is None else torch.tensor(loss_mask, device=device, dtype=terms.dtype)
    if mask.numel() != terms.numel() or not bool(mask.sum() > 0):
        raise ValueError("Invalid loss positions")
    return (terms * mask).sum(), logp


def train_step(model, reference, optimizer, scheduler, items, advantages, masks, cfg, prior_steps):
    import torch
    used = sum(sum(mask) for mask in masks)
    active = sum(sum(mask) for mask, a in zip(masks, advantages) if abs(a) > 1e-12)
    # With beta=0 there is no objective on a constant-reward batch.
    # Calling AdamW with zero gradients would still move its stored moments
    # and parameters. Keep the older, explicitly nonzero-KL route unchanged.
    do_step = active > 0 or (cfg["kl_beta"] != 0 and prior_steps > 0)
    loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    if do_step:
        disable_training_dropout(model)
        for item, adv, mask in zip(items, advantages, masks):
            if not sum(mask):
                continue
            loss, _ = token_loss(model, item["prompt_token_ids"], item["completion_token_ids"], adv,
                                 mask, reference, cfg["kl_beta"])
            (loss / used).backward()
            loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite Solver gradient: do not update or commit")
        optimizer.step()
        scheduler.step()
    return {"tokens_used": used, "nonzero_advantage_tokens": active,
            "optimizer_step": do_step, "loss": loss_sum / used}


def make_fresh_solver_branch(base, *, learning_rate):
    """Clone weights only; never inherit base optimizer or scheduler state."""
    import copy
    import math
    import torch
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    model = copy.deepcopy(base)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Solver branch has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    return model, optimizer, scheduler


def binary_solver_train_step(model, optimizer, scheduler, items, rewards, *, beta=0., prior_steps=0):
    """Operational entry: full generated actions and binary full-pass rewards.

    Token budget admission and immutable stage verifier are caller duties.
    This wrapper neither truncates completions nor applies phase loss masks.
    """
    from generated_budget import binary_advantages
    if beta != 0 or any(group["weight_decay"] != 0 for group in optimizer.param_groups):
        raise ValueError("VERGE Solver requires beta=0 and weight_decay=0")
    if len(items) != 8:
        raise ValueError("one complete eight-rollout group is required")
    advantages = binary_advantages(rewards)
    if any(not item.get("prompt_token_ids") or not item.get("completion_token_ids") for item in items):
        raise ValueError("all prompt and completion token sequences must be nonempty")
    masks = [[1] * len(item["completion_token_ids"]) for item in items]
    return train_step(model, None, optimizer, scheduler, items, advantages, masks,
                      {"kl_beta": 0.}, prior_steps)
