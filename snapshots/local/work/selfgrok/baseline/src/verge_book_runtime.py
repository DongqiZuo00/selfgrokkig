"""GPU initialization and cross-round context for manuscript execution."""
from common import EXP_ROOT, read_json
from verge_book_protocol import SUITE, compact_round


def copy_untrained_root(role, destination, source_suite):
    """Copy only the original zero-update root, never an adaptive v1 state."""
    import shutil
    import tempfile
    import torch
    source = EXP_ROOT / "checkpoints" / f"{source_suite}_initial_{role}" / "resume_u0000"
    marker = read_json(source / "verge_committed.json")
    state = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
    runtime = state["runtime_state"]
    if (marker["update"] != 0 or runtime.get("book_initialization") is not True
            or runtime.get("trained_tokens") != 0 or runtime.get("human_curriculum") is not False
            or state["optimizer"]["state"]):
        raise RuntimeError("Restart source is not the original untrained book root")
    del state
    adapter = read_json(source / "adapter_config.json")
    if adapter["r"] != 16 or adapter["lora_alpha"] != 32:
        raise RuntimeError("Unexpected original root adapter geometry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="root_copy_", dir=destination.parent)
    shutil.copytree(source, temporary, dirs_exist_ok=True)
    from pathlib import Path
    Path(temporary).rename(destination)
    from common import atomic_json
    atomic_json(destination.parent / "restart_root_provenance.json", {
        "copied_from": str(source.relative_to(EXP_ROOT)), "trained_tokens": 0,
        "optimizer_state_entries": 0, "checkpoint_hash_scan": False,
        "source_modified": False})


def initialize_roots(cfg):
    from train_branch import build_model
    from verge_round_runtime import commit, optimizer_for, free_models
    for role, lr in (("solver", cfg["solver_learning_rate"]), ("challenger", cfg["challenger_learning_rate"])):
        branch = f"{SUITE}_initial_{role}"
        path = EXP_ROOT / "checkpoints" / branch / "resume_u0000"
        if (path / "verge_committed.json").exists():
            saved = read_json(path / "adapter_config.json")
            if saved["r"] != 16 or saved["lora_alpha"] != 32:
                raise RuntimeError("Book initialization is not the frozen rank16/alpha32 root")
            continue
        if cfg.get("initial_root_copy_from_suite"):
            copy_untrained_root(role, path, cfg["initial_root_copy_from_suite"])
            continue
        tokenizer, model = build_model(42, None)
        optimizer, scheduler = optimizer_for(model, lr)
        commit(model, optimizer, scheduler, branch, 0, {
            "role": role, "book_initialization": True, "trained_tokens": 0,
            "human_curriculum": False, "teacher_steps_prior": 0})
        del tokenizer, model, optimizer, scheduler
        free_models()


def history(cfg):
    output = []
    for name in cfg["prior_book_versions"]:
        root = EXP_ROOT / "raw_results" / name
        if not (EXP_ROOT / "manifests" / f"{name}_complete.json").exists():
            raise RuntimeError("Refusing uncommitted historical round")
        frozen = read_json(root / "round_frozen.json")
        branches = {b["index"]: read_json(root / "branches" / str(b["index"]) / "complete.json")
                    for b in frozen["branches"]}
        output.append(compact_round(frozen, read_json(root / "decision.json"), branches,
                                    read_json(root / "challenger_update.json")))
    return output


def inherit_optimizer(optimizer, scheduler, path):
    import torch
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    runtime = state["runtime_state"]
    prior = runtime.get("teacher_result", {}).get("cumulative_optimizer_steps",
                runtime.get("teacher_steps_prior", 0))
    del state
    return int(prior)
