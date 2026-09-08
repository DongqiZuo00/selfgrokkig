from __future__ import annotations

import random

from datasets import load_from_disk

from common import WORK_ROOT
from prepend_conditions import required_prefix


source = load_from_disk(str(WORK_ROOT / "data" / "official_train" / "prepend_sequence_train"))
saved_train = load_from_disk(str(WORK_ROOT / "data" / "target_splits" / "target_train"))
saved_development = load_from_disk(str(WORK_ROOT / "data" / "target_splits" / "development"))
indices = list(range(len(source)))
random.Random(20260826).shuffle(indices)
reconstructed_train = source.select(indices[:480])
if [str(row["id"]) for row in reconstructed_train] != [str(row["id"]) for row in saved_train]:
    raise RuntimeError("saved PREPEND target_train differs from its source-manifest reconstruction")
selection = saved_development.select(range(32))
for row in selection:
    required_prefix(dict(row))
if {str(row["id"]) for row in saved_train} & {str(row["id"]) for row in selection}:
    raise RuntimeError("official PREPEND train and selection overlap")
print("OFFICIAL_PREPEND_SPLIT_VALID", len(saved_train), len(selection))
