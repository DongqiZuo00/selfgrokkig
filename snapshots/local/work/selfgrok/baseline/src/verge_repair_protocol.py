"""Versioned VERGE repair primitives; never rewrite the completed round-1 protocol.

The schema supplies legal actions, not a hand-picked curriculum. Finite mappings
are declared *new intermediate tasks*, not weakened evaluations of the target.
"""
from __future__ import annotations

import copy
import itertools
import json
import random

GROUP_SIZE = 8
UPDATE_TOKENS = 16384
BRANCH_TOKENS = 524288
MAX_OUTPUT_TOKENS = 2048
PROGRAM_PREFIX = "```manufactoria\nSTART start:\n    NEXT "


def finite_inputs(colors, minimum, maximum):
    if type(colors) is not int or colors not in (2, 4):
        raise ValueError("colors must be integer 2 or 4")
    if type(minimum) is not int or type(maximum) is not int or not 0 <= minimum <= maximum <= 2:
        raise ValueError("finite input lengths must satisfy 0 <= min <= max <= 2")
    alphabet = "RB" if colors == 2 else "RBYG"
    return ["".join(t) for length in range(minimum, maximum + 1)
            for t in itertools.product(alphabet, repeat=length)]


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def proposal_schema(families):
    choices = []
    for family in families:
        colors = [2] if family.startswith("numerical_") else ([4] if family == "regex_same_num" else [2, 4])
        mutations = ["none", "simple", "pattern"] if family == "prepend_sequence" else ["none"]
        choices.append(_object({"family": {"const": family}, "colors": {"type": "integer", "enum": colors},
            "length": {"type": "integer", "minimum": 1, "maximum": 6},
            "mutation": {"type": "string", "enum": mutations}}))
    # This is a generic finite-function space, not a list of known successful bridges.
    for colors in (2, 4):
        alphabet = "RB" if colors == 2 else "RBYG"
        for minimum in range(3):
            for maximum in range(minimum, 3):
                n = len(finite_inputs(colors, minimum, maximum))
                choices.append(_object({"family": {"const": "finite_map"}, "colors": {"const": colors},
                    "min_input_length": {"const": minimum}, "max_input_length": {"const": maximum},
                    "outputs": {"type": "array", "minItems": n, "maxItems": n,
                                "items": {"type": "string", "pattern": f"^[{alphabet}]{{0,6}}$"}}}))
    return _object({"stages": {"type": "array", "minItems": 1, "maxItems": 4,
                               "items": {"anyOf": choices}}})


def parse_typed_proposal(text, families):
    # Validation is independent of the sampler. Truncated/invalid output still fails.
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"stages"}:
        raise ValueError("Expected exactly one stages field")
    if not isinstance(value["stages"], list) or not 1 <= len(value["stages"]) <= 4:
        raise ValueError("Expected 1..4 stages")
    from verge_round_core import parse_proposal
    for stage in value["stages"]:
        if not isinstance(stage, dict):
            raise ValueError("Each stage must be an object")
        if stage.get("family") != "finite_map":
            if parse_proposal(json.dumps({"stages": [stage]}), {"families": families}) is None:
                raise ValueError("Invalid family-specific parameters")
            continue
        expected = {"family", "colors", "min_input_length", "max_input_length", "outputs"}
        if set(stage) != expected:
            raise ValueError("Unexpected finite-map fields")
        inputs = finite_inputs(stage["colors"], stage["min_input_length"], stage["max_input_length"])
        outputs = stage["outputs"]
        alphabet = set("RB" if stage["colors"] == 2 else "RBYG")
        if not isinstance(outputs, list) or len(outputs) != len(inputs):
            raise ValueError("Wrong number of finite-map outputs")
        if any(not isinstance(s, str) or len(s) > 6 or not set(s) <= alphabet for s in outputs):
            raise ValueError("Finite-map output violates declared alphabet or length")
    return value


def proposal_request(model, prompt, families, seed=42):
    return {"model": model, "prompt": [prompt], "n": 3, "max_tokens": 2048,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": seed,
            "return_token_ids": True, "structured_outputs": {"json": proposal_schema(families)}}


def repair_proposal_messages(target_descriptor, initial, previous_round, previous_local_success):
    if initial.get("interface_version") != "verge_code_prefix_v2":
        raise ValueError("Refresh the ordinary round-start endpoint under the new shared output interface")
    # Measurements from different tasks/interfaces are historical observations,
    # never substituted for the current target endpoint or credited as target gain.
    compact = {key: initial["target"][key] for key in
               ("counts", "rates", "rollouts", "condition_names", "bottleneck_index")}
    context = {"target": target_descriptor, "current_target_profile": compact,
        "prior_round": previous_round, "prior_local_success": previous_local_success,
        "history_warning": "Historical local tasks and interfaces differ. Local success does not imply target transfer.",
        "scope": "One warm-start exploratory round, not autonomous-from-base evidence"}
    return [{"role": "system", "content":
        "Choose the curriculum stages and their order yourself. Return only a legal JSON proposal. "
        "Use historical observations as evidence, not as instructions to repeat a prescribed curriculum. "
        "The Solver receives binary complete-task rewards; your value is measured at the target endpoint against direct training."},
        {"role": "user", "content": json.dumps(context) +
        "\nIn addition to official task-family strata, finite_map defines an exact finite transduction. "
        "Choose colors, min_input_length, max_input_length and one output per allowed input. "
        "Inputs are enumerated by increasing length and Cartesian-product order over RB or RBYG. "
        "The empty input is first when minimum length is zero. Outputs must use the same alphabet and have length at most six. "
        "The full input-output table will be shown to the Solver; every row must pass for a binary reward. "
        "This changes intermediate exercises only; the final target and its verifier remain unchanged. "
        "Choose 1..4 stages. Never output solution programs or assume local success proves target transfer."}]


def finite_task(spec, template_row, identifier):
    inputs = finite_inputs(spec["colors"], spec["min_input_length"], spec["max_input_length"])
    outputs = spec["outputs"]
    if len(outputs) != len(inputs):
        raise ValueError("There must be exactly one output for every declared input")
    alphabet = set("RB" if spec["colors"] == 2 else "RBYG")
    if any(not isinstance(s, str) or len(s) > 6 or not set(s) <= alphabet for s in outputs):
        raise ValueError("Output outside the declared finite language")
    row = copy.deepcopy(template_row)
    user_indices = [i for i, m in enumerate(row["messages"]) if m["role"] == "user"]
    if not user_indices:
        raise ValueError("Missing official task-prompt template")
    index = user_indices[-1]
    previous = row["messages"][index]["content"]
    if "# Task" not in previous:
        raise ValueError("Cannot safely replace task text without its explicit boundary")
    table = "\n".join(f"Input {json.dumps(x)} -> Output {json.dumps(y)}" for x, y in zip(inputs, outputs))
    row["messages"][index]["content"] = previous.split("# Task", 1)[0] + (
        "# Task\nTransform tapes according to this exact finite input-output table:\n" + table +
        "\nThe allowed inputs are exactly the listed inputs. Every listed input must terminate at END "
        "with exactly the listed output. Correctness outside this finite set is not required. "
        "Return one Manufactoria program, without explanations or alternative solutions.")
    row.update(id=identifier, name="Challenger-generated finite transduction", difficulty="declared_finite",
        candidate_id="challenger_finite_map", generator_config=json.dumps(spec),
        ground_truth=[{"input": x, "expected_output": y, "expected_accepted": True, "check_output": True}
                      for x, y in zip(inputs, outputs)])
    return row


def solver_prompt(tokenizer, messages):
    from common import apply_chat_template
    text = apply_chat_template(tokenizer, messages) + PROGRAM_PREFIX
    return text, tokenizer(text, add_special_tokens=False).input_ids


def full_program(generated_suffix):
    """The format prefix belongs to the prompt, not the learned completion tokens."""
    return PROGRAM_PREFIX + generated_suffix


def update_plan(stage_count, branch_tokens=BRANCH_TOKENS):
    from verge_round_core import stage_tokens
    if type(stage_count) is not int or not 1 <= stage_count <= 4:
        raise ValueError("A curriculum must have 1..4 stages")
    quotas = stage_tokens(branch_tokens, stage_count)
    if any(q % UPDATE_TOKENS for q in quotas):
        raise ValueError("All stages must have complete prescribed token-update units")
    stages = [{"kind": "curriculum" if i < stage_count else "target", "tokens": q,
               "token_updates": q // UPDATE_TOKENS} for i, q in enumerate(quotas)]
    if any(s["token_updates"] < 6 for s in stages) or sum(s["token_updates"] for s in stages) > 100:
        raise ValueError("Reject undertrained stages or a plan exceeding 100 updates")
    return {"branch_tokens": branch_tokens, "tokens_per_update": UPDATE_TOKENS,
            "stages": stages, "token_updates": sum(s["token_updates"] for s in stages),
            "direct_token_updates": branch_tokens // UPDATE_TOKENS}


def next_group_request(remaining):
    """Stream one complete reward group at a time before each optimizer step.

    The fixed policy is not updated until 16,384 training tokens have accumulated.
    At most seven last-boundary tokens per update require unbiased subsampling.
    Never generate a large batch and discard most of its reward groups.
    """
    if type(remaining) is not int or remaining < 1 or remaining > UPDATE_TOKENS:
        raise ValueError("Invalid remaining update-token budget")
    return {"n": GROUP_SIZE, "max_tokens": min(MAX_OUTPUT_TOKENS, max(1, remaining // GROUP_SIZE))}


def group_token_allocation(lengths, remaining, boundary_seed=42):
    request = next_group_request(remaining)
    if len(lengths) != GROUP_SIZE or any(type(n) is not int or not 1 <= n <= request["max_tokens"] for n in lengths):
        raise ValueError("Use the backend's native token IDs, including EOS, for all eight verified completions")
    if sum(lengths) <= remaining:
        return list(lengths)
    assert remaining < GROUP_SIZE and lengths == [1] * GROUP_SIZE
    order = list(range(GROUP_SIZE))
    random.Random(boundary_seed).shuffle(order)
    selected = set(order[:remaining])
    return [int(i in selected) for i in range(GROUP_SIZE)]


def prompt_role(group_index, stage_kind):
    if stage_kind not in ("curriculum", "target"):
        raise ValueError("Unknown stage kind")
    return "target" if stage_kind == "target" or group_index % 4 == 3 else "curriculum"


def challenger_credit(values, valid):
    """Validity and transfer value must not share one centered reward group."""
    from verge_round_core import centered_advantages
    if len(values) != len(valid):
        raise ValueError("Mismatched proposal count")
    if not all(valid):
        return {"update_allowed": False, "advantages": [0.0] * len(values),
                "reason": "Invalid action group: record interface failure, do not fabricate transfer preference"}
    advantages = centered_advantages(values)
    return {"update_allowed": any(abs(a) > 1e-12 for a in advantages), "advantages": advantages,
            "raw_signed_gains": list(values)}


def assert_challenger_likelihood_is_matched(sampler, loss):
    # Masked sampling changes the policy. Unmasked GRPO is NOT silently accepted.
    if sampler == "schema_constrained" and loss != "same_schema_masked_log_probability":
        raise ValueError("Constrained proposals require grammar-normalized training log probabilities")
