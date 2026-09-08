"""Build VERGE Evaluation objects from complete per-rollout raw evidence.

This module never generates a completion and never invents a test grouping.
The caller supplies actual frozen verifier condition bits and static cycle facts.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from verge_protocol import CONDITION_COUNT, Candidate, Evaluation


@dataclass(frozen=True)
class RawRollout:
    checkpoint: str
    training_seed: int
    manifest_id: str
    instance_id: str
    sample_id: str
    conditions: tuple[bool, ...]
    has_cycle: bool | None

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.checkpoint, self.manifest_id, self.instance_id, self.sample_id)):
            raise ValueError("raw evidence needs nonempty checkpoint/manifest/instance/sample identities")
        if type(self.training_seed) is not int:
            raise ValueError("training_seed must identify one integer training seed")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if len(self.conditions) != CONDITION_COUNT or any(type(x) is not bool for x in self.conditions):
            raise ValueError("raw evidence must contain six binary bool condition results")
        if any(a < b for a, b in zip(self.conditions, self.conditions[1:])):
            raise ValueError("raw condition results must be nested")
        if self.conditions[0]:
            if type(self.has_cycle) is not bool:
                raise ValueError("parse-valid programs require an explicit static cycle fact")
        elif self.has_cycle is not None:
            raise ValueError("unparsed programs have no static cycle fact; use None")


@dataclass(frozen=True)
class EvidenceBatch:
    evaluation: Evaluation
    cycle_count: int
    raw_records: tuple[RawRollout, ...]

    @property
    def candidate(self):
        return Candidate(self.evaluation, self.cycle_count)

    def first_eight(self):
        """Derive P8 from these very same P32 records; perform no re-sampling."""
        evaluation = self.evaluation
        if len(evaluation.sample_ids) != 32:
            raise ValueError("first_eight must be derived from a full 32-sample raw batch")
        sample_ids = evaluation.sample_ids[:8]
        records = tuple(record for record in self.raw_records if record.sample_id in sample_ids)
        return aggregate_rollouts(records, checkpoint=evaluation.checkpoint,
                                  training_seed=evaluation.training_seed,
                                  manifest_id=evaluation.manifest_id,
                                  instance_ids=evaluation.instance_ids,
                                  sample_ids=sample_ids)


def aggregate_rollouts(records: Sequence[RawRollout | Mapping], *, checkpoint: str,
                       training_seed: int, manifest_id: str,
                       instance_ids: Sequence[str], sample_ids: Sequence[str]) -> EvidenceBatch:
    """Require exact coverage of frozen instance x sample IDs, then aggregate.

    One record per instance/sample is mandatory, including failed generations.
    Fixed sample count must be 8 or 32. Raw input ordering is irrelevant; returned
    records use the frozen instance/sample ordering. An explicit seed is checked
    on every record and never pooled as an extra evaluation instance.
    """
    instance_ids, sample_ids = tuple(instance_ids), tuple(sample_ids)
    if not instance_ids or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("unique expected selection instances required")
    if len(sample_ids) not in (8, 32) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("exactly 8 or 32 distinct frozen sample IDs required")
    expected = {(instance, sample) for instance in instance_ids for sample in sample_ids}
    indexed = {}
    for supplied in records:
        record = supplied if isinstance(supplied, RawRollout) else RawRollout(**supplied)
        if (record.checkpoint, record.training_seed, record.manifest_id) != (checkpoint, training_seed, manifest_id):
            raise ValueError("raw evidence mixes checkpoint, training seed, or selection manifest")
        key = record.instance_id, record.sample_id
        if key not in expected or key in indexed:
            raise ValueError("unexpected or duplicated instance/sample raw record")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("raw evidence must include every frozen instance/sample, including failures")
    canonical = tuple(indexed[(instance, sample)] for instance in instance_ids for sample in sample_ids)
    counts = tuple(tuple(sum(indexed[(instance, sample)].conditions[j] for sample in sample_ids)
                         for j in range(CONDITION_COUNT)) for instance in instance_ids)
    evaluation = Evaluation(checkpoint, training_seed, manifest_id, instance_ids, sample_ids, counts)
    return EvidenceBatch(evaluation, sum(record.has_cycle is True for record in canonical), canonical)
