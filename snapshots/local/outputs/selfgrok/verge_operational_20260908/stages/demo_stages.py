"""Synthetic, deterministic CPU smoke run; zero model rollouts or training."""
import json

from operational_stages import RejectionSamplingBuffer, validate_stage
from test_operational_stages import PROTECTED, REGISTRY, add_round, library, probe, stage


def main():
    validations = {}
    for kind in ("registered", "tape_transform", "hinted_target"):
        extra = {"library": library(), "hint_probe": probe()} if kind == "hinted_target" else {}
        validations[kind] = validate_stage(stage(kind), registry=REGISTRY,
                                           protected_tuples=PROTECTED, **extra).to_dict()
    buffer = RejectionSamplingBuffer(n_min=2)
    add_round(buffer, seed=1)
    assert buffer.plan_update() is None
    add_round(buffer, seed=2)
    plan = buffer.plan_update()
    assert plan and len(plan["cumulative_examples"]) == 2
    assert all(row["accepted"] for row in validations.values())
    print(json.dumps({"synthetic_fixture": True, "formal_model_training": False,
                      "model_rollouts_executed": 0, "validations": validations,
                      "rft_plan_only": plan, "buffer": buffer.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
