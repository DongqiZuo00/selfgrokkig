"""Fail-closed runtime authorization for separately frozen direct-control blocks.

No defaults choose a scientific warm-up, no submission, and no model sampling.
The block dispatcher is responsible for resource-safe dependency ownership.
"""
import json
from pathlib import Path
from verge_control_protocol import segment_config, validate_control_config

LABELS = ("binary", "dense", "dense_then_binary")


def validate_block(block):
    if (block.get("protocol") != "verge_control_v1_direct_block"
            or block.get("labels") != list(LABELS)
            or block.get("segment_loss_tokens") != [524288]*6
            or block.get("training_seed") != 42
            or block.get("maximum_total_gpus") != 2
            or block.get("maximum_total_cpu_memory_gb") != 64
            or block.get("requires_primary_complete") is not True
            or block.get("optimizer_reset") != "fresh_each_segment"
            or block.get("total_compute_matched") is not False):
        raise ValueError("Unsupported or enlarged direct-control comparison")
    warmup = block.get("dense_warmup_loss_tokens")
    if type(warmup) is not int or not 0 < warmup < sum(block["segment_loss_tokens"]):
        raise ValueError("An explicit interior GLOBAL loss-token warm-up is required")
    return block


def phases_for(block, label, round_index):
    validate_block(block)
    if label not in LABELS or type(round_index) is not int or not 0 <= round_index < 6:
        raise ValueError("Unknown control arm or segment")
    budget = block["segment_loss_tokens"][round_index]
    if label != "dense_then_binary":
        return [{"kind":"target", "reward_mode":label, "tokens":budget}]
    before = sum(block["segment_loss_tokens"][:round_index])
    dense = max(0, min(budget, block["dense_warmup_loss_tokens"]-before))
    return [{"kind":"target", "reward_mode":mode, "tokens":tokens}
            for mode, tokens in (("dense",dense), ("binary",budget-dense)) if tokens]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_launch(exp, cfg):
    """Reject missing/unfrozen plans, configuration drift and uncommitted ancestry."""
    exp = Path(exp).resolve()
    validate_control_config(cfg, cfg["protocol_version"])
    source_plan = exp / cfg["control_block_protocol"]
    frozen_plan = exp / "raw_results/verge_control_v1/protocol_frozen.json"
    if not source_plan.is_file() or not frozen_plan.is_file():
        raise RuntimeError("No separately frozen direct-control block; training is not enabled")
    block = validate_block(read(source_plan))
    if read(frozen_plan) != block:
        raise RuntimeError("Direct-control block differs from its submitted frozen copy")
    primary = exp / "raw_results/verge_book_v2/PRIMARY_BLOCK_COMPLETE.json"
    primary_record = read(primary) if primary.is_file() else {}
    if (primary_record.get("primary_block_complete") is not True
            or primary_record.get("outer_rounds") != 24):
        raise RuntimeError("Do not overlap new control dispatch with the running primary block")
    label, r = cfg["control_label"], cfg["control_round"]
    expected = segment_config(exp, label, r, phases=phases_for(block,label,r),
        starting_checkpoint=cfg["solver_start"], block_protocol=cfg["control_block_protocol"])
    if cfg != expected:
        raise RuntimeError("Segment configuration differs from the frozen comparison plan")
    source = (exp / cfg["solver_start"]).resolve()
    if (not source.is_relative_to((exp / "checkpoints").resolve())
            or not (source / "verge_committed.json").is_file()
            or not (source / "adapter_model.safetensors").is_file()
            or not (source / "state.pt").is_file()):
        raise RuntimeError("Missing, foreign or uncommitted Solver source")
    marker = read(source / "verge_committed.json")
    if marker.get("update") != int(source.name.removeprefix("resume_u")):
        raise RuntimeError("Checkpoint marker does not match the source update")
    if r:
        prior = exp / "raw_results" / f"verge_control_v1_{label}_r{r-1:02d}" / "branches/0/complete.json"
        if not prior.is_file():
            raise RuntimeError("Previous same-arm control segment has not completed")
        record = read(prior)
        if (record.get("control_segment_verified") is not True
                or record.get("control_label") != label or record.get("checkpoint") != cfg["solver_start"]
                or record["training"]["train_tokens"] != block["segment_loss_tokens"][r-1]):
            raise RuntimeError("Previous control segment does not certify this continuation")
    return block


def prepared_round(cfg, initial):
    validate_control_config(cfg, cfg["protocol_version"])
    if initial["checkpoint"] != cfg["solver_start"]:
        raise ValueError("Initial endpoint belongs to another checkpoint")
    return {"config":cfg, "initial":initial, "control_only":True,
            "generation":None, "new_curricula":0,
            "branches":[{"index":0, "id":cfg["protocol_version"]+"_direct",
                         "stages":[dict(p, path=cfg["target_train"]) for p in cfg["control_phases"]]}]}


def validate_prepared(cfg, frozen):
    if frozen != prepared_round(cfg, frozen["initial"]):
        raise RuntimeError("Prepared control segment differs from its frozen configuration")
    return frozen
