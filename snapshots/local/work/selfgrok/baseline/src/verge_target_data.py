"""Build the paper's fixed target through the official generator, without model auditing."""
import json

from datasets import Dataset, load_from_disk
from common import EXP_ROOT, atomic_json, seed_everything
from prepare_data import GeneratorRegistry, GeneratorConfig, TrainingFileWrapper
from verge_round_core import config, DATA, VERSION

PARAMETERS = {"prepend_sequence": "RBYGRB", "mutation_type": "replace_pattern_to_pattern",
              "source_pattern": "RB", "target_pattern": "YG"}


def validate(rows):
    for row in rows:
        assert json.loads(row["generator_config"])["official_parameter"] == PARAMETERS
        for case in row["ground_truth"]:
            assert case["check_output"] and case["expected_accepted"]
            assert case["expected_output"] == "RBYGRB" + case["input"].replace("RB", "YG")


def prepare_target():
    cfg = config()
    generator = GeneratorRegistry.get_generator("prepend_sequence")
    wrapper = TrainingFileWrapper()
    counts = {}
    for role, count, offset in (("target_train", 128, 0), ("target_selection", 64, 1000),
                                ("target_completion", 64, 2000)):
        path = EXP_ROOT / cfg[role]
        if path.exists():
            generated = [dict(r) for r in load_from_disk(str(path))]
        else:
            generated = []
            for i in range(count):
                seed_everything(42 + offset + i)
                problem = generator.generate_problem(GeneratorConfig.FOUR_COLOR_CHARS,
                    is_four_color=True, params=dict(PARAMETERS), index=i)
                problem.update(id=f"{VERSION}_{role}_{i:04d}", difficulty_level="fixed_primary",
                               pattern_type="prepend_sequence", color_mode="four_color")
                row = wrapper.convert_problem(problem)
                row["candidate_id"] = "verge_fixed_primary"
                row["generator_config"] = json.dumps({"official_parameter": PARAMETERS, "split": role})
                generated.append(row)
            validate(generated)
            Dataset.from_list(generated).save_to_disk(str(path))
        assert len(generated) == count
        validate(generated)
        counts[role] = count
    atomic_json(DATA / "target_contract.json", {"parameters": PARAMETERS, "counts": counts,
        "distinct_task_definitions": 1, "note": "Generator instances/samples are not independent task families.",
        "official_test_opened": False})


if __name__ == "__main__":
    prepare_target()
