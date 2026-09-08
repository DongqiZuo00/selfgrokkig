"""Versioned backend with disjoint per-prompt vLLM child RNG ranges.

Used only by explicitly enabled recovery runs; legacy caches remain unchanged.
Distinct configured PRNG streams do not guarantee distinct sampled strings.
"""
from concurrent.futures import ThreadPoolExecutor
import time

PROTOCOL = "per_prompt_disjoint_seed_ranges_v1"


def split_requests(request):
    prompts = request["prompt"]
    n, base = request["n"], request["seed"]
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, list) or not p for p in prompts):
        raise ValueError("Expected nonempty native-token prompt lists")
    if type(n) is not int or n < 1 or type(base) is not int or base < 0:
        raise ValueError("Explicit nonnegative seed and positive n are required")
    if base + len(prompts) * n - 1 >= 2**63:
        raise ValueError("Independent child seeds exceed the supported integer range")
    # vLLM assigns seed + draw_index to each of a prompt's n completions.
    # Offset each prompt by n, so no two (prompt, draw) pairs share a seed.
    return [dict(request, prompt=[prompt], seed=base + i * n) for i, prompt in enumerate(prompts)]


def merge_responses(request, responses):
    parts = split_requests(request)
    if len(parts) != len(responses):
        raise ValueError("Missing per-prompt response")
    choices = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for i, (part, response) in enumerate(zip(parts, responses)):
        group = sorted(response["choices"], key=lambda c: c["index"])
        if [c["index"] for c in group] != list(range(request["n"])):
            raise ValueError("Incomplete or duplicate per-prompt choice indices")
        if response.get("model", request["model"]) != request["model"]:
            raise ValueError("A response came from another policy")
        for c in group:
            if not c.get("token_ids"):
                raise ValueError("Native action IDs are required")
            choices.append(dict(c, index=i * request["n"] + c["index"]))
        for key in usage:
            if key not in response.get("usage", {}):
                raise ValueError("Native usage accounting is missing")
            usage[key] += response["usage"][key]
    return {"object": "text_completion", "model": request["model"], "choices": choices,
        "usage": usage, "backend_protocol": PROTOCOL,
        "prompt_parent_seeds": [part["seed"] for part in parts],
        "child_seed_count": len(parts) * request["n"],
        "original_response_ids": [r.get("id") for r in responses]}


def validate_response(request, response):
    parts = split_requests(request)
    expected = len(parts) * request["n"]
    if (response.get("backend_protocol") != PROTOCOL
            or response.get("prompt_parent_seeds") != [p["seed"] for p in parts]
            or response.get("child_seed_count") != expected
            or sorted(c["index"] for c in response["choices"]) != list(range(expected))):
        raise RuntimeError("Endpoint does not certify disjoint configured child RNG streams")


def validate_records(request, records):
    expected = len(request["prompt"]) * request["n"]
    if len(records) != expected:
        raise RuntimeError("Incomplete independently sampled batch")
    for i, record in enumerate(records):
        prompt_index, draw = divmod(i, request["n"])
        if (record.get("sampling_protocol") != PROTOCOL
                or record.get("sampling_parent_seed") != request["seed"] + prompt_index * request["n"]
                or record.get("sampling_child_seed") != request["seed"] + i
                or record.get("instance_index") != prompt_index
                or record.get("rollout_index") != draw
                or record.get("prompt_token_ids") != request["prompt"][prompt_index]):
            raise RuntimeError("Rollout RNG/prompt mapping does not match the frozen batch")


def complete(base_url, request, parts_dir, http_json, read_json, atomic_json, max_workers=8):
    """Bounded concurrent requests with exact-request per-prompt recovery caches."""
    if not 1 <= max_workers <= 8:
        raise ValueError("At most eight outstanding HTTP requests are permitted")
    parts = split_requests(request)

    def fetch(item):
        i, part = item
        path = parts_dir / f"prompt_{i:04d}.json"
        if path.exists():
            saved = read_json(path)
            if saved["request"] != part:
                raise RuntimeError("Per-prompt cache belongs to a different request/policy")
            return saved, True
        start = time.perf_counter()
        response = http_json(base_url + "/v1/completions", part, timeout=3600)
        # Check the response before treating it as a completed cache entry.
        merge_responses(part, [response])
        saved = {"request": part, "response": response, "seconds": time.perf_counter() - start}
        atomic_json(path, saved)
        return saved, False

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(parts))) as pool:
        results = list(pool.map(fetch, enumerate(parts)))
    response = merge_responses(request, [saved["response"] for saved, _ in results])
    response["recovery_accounting"] = {
        "cached_prompt_requests": sum(cached for _, cached in results),
        "new_prompt_requests": sum(not cached for _, cached in results),
        "sum_request_seconds_including_recovered_requests": sum(saved["seconds"] for saved, _ in results),
        "wall_seconds_this_invocation": time.perf_counter() - start}
    return response
