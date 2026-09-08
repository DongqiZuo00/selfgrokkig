"""Native-token, fixed-cap sampling. No retokenization of sampled actions."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from common import atomic_json, read_json, read_jsonl, stable_int, extract_program
from vllm_runtime import http_json, PROGRAM_STOPS, TelemetrySampler, _verify_payload
from verge_repair_protocol import solver_prompt, full_program


def atomic_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require_native_choice(choice, expected_prompt):
    ids = choice.get("token_ids")
    if not ids or any(type(i) is not int or i < 0 for i in ids):
        raise RuntimeError("Missing native sampled token IDs; retokenization is forbidden")
    actual = choice.get("prompt_token_ids")
    if actual is not None and actual != expected_prompt:
        raise RuntimeError("Server changed the exact learner prompt token IDs")
    return ids


def score_rows(client, rows, samples, seed, path, telemetry_path=None, sampling_seed_key=None,
               independent_prompt_streams=False):
    from verge_independent_sampling import PROTOCOL, complete, validate_response, validate_records
    prompts = [solver_prompt(client.tokenizer, row["messages"])[1] for row in rows]
    request = {"model": client.model_name, "prompt": prompts, "n": samples,
        "max_tokens": 2048, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "seed": stable_int(seed, sampling_seed_key or "verge_v2"),
        "return_token_ids": True, "stop": PROGRAM_STOPS, "include_stop_str_in_output": True}
    response_path = path.with_suffix(".response.json")
    backend = PROTOCOL if independent_prompt_streams else None
    if response_path.exists():
        stored = read_json(response_path)
        if stored["request"] != request or stored.get("backend_protocol") != backend:
            raise RuntimeError(f"Cached action batch has a different policy/request: {path}")
        response, elapsed = stored["response"], stored["seconds"]
    else:
        started = time.perf_counter()
        with TelemetrySampler(client.base_url, client.gpu, telemetry_path or path.with_suffix(".telemetry.jsonl")):
            if independent_prompt_streams:
                response = complete(client.base_url, request, path.with_suffix(".parts"),
                                    http_json, read_json, atomic_json)
            else:
                response = http_json(client.base_url + "/v1/completions", request, timeout=3600)
        elapsed = time.perf_counter() - started
        stored = {"request": request, "response": response, "seconds": elapsed}
        if independent_prompt_streams:
            validate_response(request, response)
            stored["backend_protocol"] = backend
        atomic_json(response_path, stored)
    if independent_prompt_streams:
        validate_response(request, response)
    expected = len(rows) * samples
    if path.exists():
        cached = read_jsonl(path)
        if len(cached) != expected or any(r.get("interface_version") != "verge_code_prefix_v2" for r in cached):
            raise RuntimeError(f"Incomplete or incompatible rollout file: {path}")
        if independent_prompt_streams:
            validate_records(request, cached)
        return cached
    choices = sorted(response["choices"], key=lambda c: c["index"])
    if [c["index"] for c in choices] != list(range(expected)):
        raise RuntimeError("Incomplete or reordered native response")
    provisional, payloads = [], []
    for i, choice in enumerate(choices):
        index, draw = divmod(i, samples)
        native = require_native_choice(choice, prompts[index])
        if len(native) > 2048:
            raise RuntimeError("Backend violated the fixed completion cap")
        completion = full_program(choice["text"])
        provisional.append({"instance_index": index, "instance_id": rows[index]["id"],
            "rollout_index": draw, "completion": completion, "sampled_suffix": choice["text"],
            "program": extract_program(completion), "prompt_token_ids": prompts[index],
            "completion_token_ids": native, "completion_tokens": len(native),
            "finish_reason": choice.get("finish_reason"), "stop_reason": choice.get("stop_reason"),
            "interface_version": "verge_code_prefix_v2", "max_tokens": 2048,
            "generation_seconds_share": elapsed / expected, "request_wall_seconds": elapsed})
        if independent_prompt_streams:
            provisional[-1].update(sampling_protocol=backend,
                sampling_parent_seed=request["seed"] + index * samples,
                sampling_child_seed=request["seed"] + i)
        payloads.append((completion, rows[index]["ground_truth"]))
    with ProcessPoolExecutor(max_workers=client.verifier_workers) as executor:
        verdicts = list(executor.map(_verify_payload, payloads, chunksize=8))
    completed = [dict(item, **verdict) for item, verdict in zip(provisional, verdicts)]
    if independent_prompt_streams:
        validate_records(request, completed)
    atomic_jsonl(path, completed)
    print(json.dumps({"batch": str(path), "rollouts": expected,
        "native_tokens": sum(r["completion_tokens"] for r in completed), "seconds": elapsed,
        "successes": sum(r["reward"] for r in completed), "sampling_protocol": backend,
        "configured_child_streams": response.get("child_seed_count")}), flush=True)
    return completed
