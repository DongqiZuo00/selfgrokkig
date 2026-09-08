"""Endpoint-only direct-control summaries; no training, sampling or sealed tests.

This produces intermediate JSON for later reporting. It never marks the control
comparison or experiment book complete and never changes continuation decisions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import verge_control_controller as ctl
from verge_control_block import validate_block
from verge_independent_sampling import PROTOCOL, validate_records, validate_response
from verge_round_core import condition_names


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verified_endpoint(directory, kind, summary, instances, draws):
    """Check saved native requests and aggregate profiles against their rows."""
    records = list(rows(directory / (kind+".jsonl")))
    saved = ctl.read(directory / (kind+".response.json"))
    request = saved["request"]
    if (saved.get("backend_protocol") != PROTOCOL or request["n"] != draws
            or len(request["prompt"]) != instances):
        raise RuntimeError("Wrong endpoint dimensions or sampler")
    validate_response(request, saved["response"])
    validate_records(request, records)
    keys = [(str(r["instance_id"]), r["rollout_index"]) for r in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate endpoint identity")
    for i in range(instances):
        if len({r["instance_id"] for r in records[i*draws:(i+1)*draws]}) != 1:
            raise RuntimeError("A prompt group contains different instances")
    for item in records:
        passed, total = item["passed_cases"], item["total_cases"]
        if (type(passed) is not int or type(total) is not int or not 0 <= passed <= total or total < 1
                or item["reward"] != int(passed == total)
                or not 0 < item["completion_tokens"] == len(item["completion_token_ids"]) <= 2048
                or item["max_tokens"] != 2048):
            raise RuntimeError("Invalid raw full-verifier outcome or action count")
    if kind == "target":
        names = condition_names()
        expected_keys = [f"{key}::{draw}" for key, draw in keys]
        if (summary["condition_names"] != names or set(summary["keyed_vectors"]) != set(expected_keys)
                or summary["rollouts"] != len(records)):
            raise RuntimeError("Target profile does not describe this endpoint")
        matrix = np.asarray([summary["keyed_vectors"][key] for key in expected_keys])
        if (matrix.shape != (len(records),len(names)) or not np.isin(matrix,[0,1]).all()
                or not (matrix[:,1:] <= matrix[:,:-1]).all()
                or not np.array_equal(matrix[:,-1],[r["reward"] for r in records])
                or not np.array_equal(matrix.sum(axis=0),summary["counts"])
                or not np.allclose(matrix.mean(axis=0),summary["rates"],rtol=0,atol=1e-12)):
            raise RuntimeError("Non-nested, inconsistent or stale target profile")
    elif kind == "scope":
        names = ["full_pass_rate","mean_case_fraction"]
        matrix = np.asarray([[r["reward"],r["passed_cases"]/r["total_cases"]] for r in records])
        if not np.allclose(matrix.mean(axis=0),[summary[k] for k in names],rtol=0,atol=1e-12):
            raise RuntimeError("Scope summary differs from its raw endpoint")
    else:
        raise ValueError("Only normal target and scope endpoints are allowed")
    # Pair identities explicitly include the *actual* prompt and child stream.
    # Never infer pairing between different arms merely from array position.
    identity = [(r["instance_id"],r["instance_index"],r["rollout_index"],
                 tuple(r["prompt_token_ids"]),r["sampling_parent_seed"],r["sampling_child_seed"])
                for r in records]
    return {"matrix":matrix.reshape(instances,draws,len(names)).astype(float),
            "identity":identity,"names":names,"records":records}


def paired_contrast(candidate, reference, replicates=2000):
    if candidate["identity"] != reference["identity"] or candidate["names"] != reference["names"]:
        raise RuntimeError("Cannot claim a paired contrast for different prompts or random streams")
    delta = candidate["matrix"] - reference["matrix"]
    if delta.shape != reference["matrix"].shape or delta.ndim != 3:
        raise RuntimeError("Unmatched endpoint dimensions")
    if type(replicates) is not int or replicates < 2:
        raise ValueError("Bootstrap replicate count must be explicit")
    n, draws, dimensions = delta.shape
    rng = np.random.default_rng(42)  # Analysis reproducibility only, not a training restart.
    estimates = np.empty((replicates,dimensions))
    # Two levels: prompt slots, then paired draws within each resampled slot.
    # Batches cap temporary memory; never load model weights or pooled trajectories.
    for offset in range(0,replicates,100):
        size = min(100,replicates-offset)
        outer = rng.integers(n,size=(size,n))
        inner = rng.integers(draws,size=(size,n,draws))
        estimates[offset:offset+size] = delta[outer[:,:,None],inner,:].mean(axis=(1,2))
    lower, upper = np.quantile(estimates,[.025,.975],axis=0)
    return [{"metric":name,"difference":float(delta[:,:,j].mean()),
             "paired_95_interval":[float(lower[j]),float(upper[j])]} for j,name in enumerate(candidate["names"])]


def generation_cost(directory, result):
    """Count unique committed rollout files once, including evaluation overhead."""
    train = result["training"]
    ledger = {"loss_tokens":train["train_tokens"],"nonzero_advantage_tokens":train["nonzero_advantage_tokens"],
              "generated_training_tokens":0,"generated_initial_endpoint_tokens":0,
              "generated_final_endpoint_tokens":0,"raw_rollouts":0,"binary_verifier_test_calls":0}
    groups = [("generated_initial_endpoint_tokens",[directory / "initial" / (k+".jsonl") for k in ("target","scope")]),
              ("generated_final_endpoint_tokens",[directory / "branches/0" / (k+".jsonl") for k in ("target","scope")]),
              ("generated_training_tokens",[directory / "branches/0/train" / f"update_{i:04d}.jsonl"
                                           for i in range(1,train["update"]+1)])]
    for name, paths in groups:
        for path in paths:
            for row in rows(path):
                ledger[name] += row["completion_tokens"]
                ledger["raw_rollouts"] += 1
                ledger["binary_verifier_test_calls"] += row["total_cases"]
    if ledger["generated_training_tokens"] != train["generated_tokens"] or train["generated_tokens"] < ledger["loss_tokens"]:
        raise RuntimeError("Training generation ledger disagrees with the committed state")
    ledger["generated_completion_tokens_all_phases"] = sum(ledger[k] for k,_ in groups)
    ledger["non_loss_generation_tokens"] = ledger["generated_completion_tokens_all_phases"]-ledger["loss_tokens"]
    ledger["challenger_generated_tokens"] = 0
    return ledger


def collect():
    root = ctl.ROOT
    block = validate_block(ctl.read(root / "protocol_frozen.json"))
    if block != ctl.read(ctl.EXP / ctl.PLAN):
        raise RuntimeError("Control plan drift; no comparison output")
    if block.get("bootstrap_replicates") != 2000:
        raise RuntimeError("Use the pre-execution analysis replicate count")
    # An analysis import or a partial segment must never look like a finished block.
    results = {(label,r):ctl.completed(label,r,deep=True) for r in range(6) for label in ctl.LABELS}
    if not all(results.values()):
        raise RuntimeError("All eighteen verified direct-control segments are required")
    endpoint_rows, comparisons, costs, trajectories = [], [], [], []
    for r in range(6):
        endpoints = {}
        for label in ctl.LABELS:
            name = ctl.version(label,r)
            directory = ctl.EXP / "raw_results" / name
            frozen = ctl.read(directory / "round_frozen.json")
            result = results[label,r]
            for endpoint_kind,summary,path in (("initial",frozen["initial"],directory / "initial"),
                                                ("final",result,directory / "branches/0")):
                # Stored endpoint.json is written only for that exact checkpoint.
                stored = ctl.read(path / "endpoint.json")
                if any(stored[k] != summary[k] for k in ("checkpoint","target","scope")):
                    raise RuntimeError("Mixed-checkpoint endpoint record")
                for kind,instances,draws in (("target",64,8),("scope",32,4)):
                    data = verified_endpoint(path,kind,summary[kind],instances,draws)
                    if endpoint_kind == "final":
                        endpoints[label,kind] = data
                    for j,metric in enumerate(data["names"]):
                        endpoint_rows.append({"arm":label,"round":r,"endpoint":endpoint_kind,
                            "checkpoint":summary["checkpoint"],"kind":kind,"metric":metric,
                            "observed_rate":float(data["matrix"][:,:,j].mean()),"rollouts":instances*draws,
                            "loss_tokens_before_endpoint":sum(block["segment_loss_tokens"][:r+(endpoint_kind == "final")])})
            train = result["training"]
            trajectories.append({"arm":label,"round":r,"updates":train["update"],
                "optimizer_steps":train["optimizer_steps"],"training_full_successes":train["target_successes"],
                "first_training_full_success":train["first_target_success"],
                "scope_safe_vs_start_observed":result["scope_safe_vs_start"],
                "note":"Training event only; never pooled into endpoint pass-rate estimates"})
            costs.append(dict(arm=label,round=r,**generation_cost(directory,result)))
        for candidate,reference in (("dense","binary"),("dense_then_binary","binary"),("dense_then_binary","dense")):
            for kind in ("target","scope"):
                for stat in paired_contrast(endpoints[candidate,kind],endpoints[reference,kind],block["bootstrap_replicates"]):
                    comparisons.append(dict(round=r,candidate=candidate,reference=reference,kind=kind,**stat))
    return {"protocol":"verge_control_v1_endpoint_analysis","segments":18,"endpoint_rows":endpoint_rows,
            "dense_warmup_loss_tokens":block["dense_warmup_loss_tokens"],
            "bootstrap_replicates":block["bootstrap_replicates"],
            "within_control_contrasts":comparisons,"trajectory_events":trajectories,"generation_ledger":costs,
            "control_block_complete":False,"suite_complete":False,"official_test_opened":False,
            "new_model_samples":0,"cross_primary_paired_comparisons":False,"total_compute_matched":False,
            "statistical_note":"Fixed-checkpoint endpoints only; exploratory two-level paired percentile intervals, no multiplicity correction or cross-training-seed inference. A zero-width empirical bootstrap interval at zero observed successes is not an upper bound on the true success probability.",
            "cost_note":"Count committed unique rollout files, not cache copies. Hardware/recovery allocation costs remain in the separate Slurm ledger; these token counts are not FLOPs. Condition reexecution is excluded from binary-verifier test counts.",
            "pending":"Final resource receipt, comparison artifacts and remaining experiment-book matrix are not completed by this JSON"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save",action="store_true")
    args = parser.parse_args()
    result = collect()
    if args.save:
        ctl.write(ctl.ROOT / "endpoint_analysis.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in (
        "endpoint_rows","within_control_contrasts","trajectory_events","generation_ledger")},indent=2))
