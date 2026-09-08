"""Inspect input schema for the new round; performs no sampling or screening."""
from collections import Counter
from datasets import load_from_disk
from common import EXP_ROOT
import inspect
from prepare_data import GeneratorRegistry

for name in ("target_train", "selection"):
    d = load_from_disk(str(EXP_ROOT / "data/self_grok_decisive_mistral_v4_2" / name))
    print(name, "rows", len(d), "test_counts", dict(Counter(len(r["ground_truth"]) for r in d)))
    print("example_names", [d[i]["name"] for i in range(min(3, len(d)))])
g = GeneratorRegistry.get_generator("prepend_sequence")
print("GENERATOR_FILE", inspect.getfile(type(g)))
print(inspect.getsource(g.generate_parameters))
print(inspect.getsource(g.generate_problem))
