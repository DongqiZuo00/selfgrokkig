"""Synthetic raw-evidence fixtures, never model outputs or benchmark classes."""

import unittest
from dataclasses import asdict, replace

from round_evidence import RawRollout, aggregate_rollouts


INSTANCES = ("fixture_i0", "fixture_i1")
SAMPLES = tuple(f"r{i}" for i in range(32))


def raw_fixture():
    records = []
    for instance in INSTANCES:
        for i, sample in enumerate(SAMPLES):
            parsed = i % 5 != 0
            conditions = (parsed, parsed, parsed and i < 20, parsed and i < 16, parsed and i < 10, parsed and i < 4)
            records.append(RawRollout("ckpt", 7, "SYNTHETIC", instance, sample, conditions, bool(i % 2) if parsed else None))
    return records


def aggregate(records, sample_ids=SAMPLES):
    return aggregate_rollouts(records, checkpoint="ckpt", training_seed=7,
                              manifest_id="SYNTHETIC", instance_ids=INSTANCES, sample_ids=sample_ids)


class RawEvidenceTests(unittest.TestCase):
    def test_complete_raw_evidence_aggregates_conditions_and_static_cycles(self):
        batch = aggregate(raw_fixture())
        self.assertEqual(batch.evaluation.counts, ((25, 25, 16, 12, 8, 3),) * 2)
        self.assertEqual(batch.cycle_count, 26)
        self.assertEqual(batch.evaluation.parse_valid_count, 50)
        self.assertEqual(batch.candidate.structure, 1)

    def test_final_first_eight_is_derived_from_identical_raw_records(self):
        final = aggregate(raw_fixture())
        probe = final.first_eight()
        self.assertEqual(probe.evaluation.sample_ids, SAMPLES[:8])
        self.assertEqual(probe.evaluation.counts, ((6, 6, 6, 6, 6, 3),) * 2)
        self.assertEqual(probe.cycle_count, 6)
        self.assertTrue(all(any(record is source for source in final.raw_records) for record in probe.raw_records))
        with self.assertRaises(ValueError):
            probe.first_eight()

    def test_input_order_and_mapping_format_do_not_change_aggregation(self):
        expected = aggregate(raw_fixture())
        supplied = [asdict(record) for record in reversed(raw_fixture())]
        actual = aggregate(supplied)
        self.assertEqual(actual, expected)

    def test_missing_duplicate_or_unexpected_instance_sample_rejected(self):
        records = raw_fixture()
        variants = [records[:-1], records + records[:1],
                    [replace(records[0], instance_id="extra")] + records[1:],
                    [replace(records[0], sample_id="extra")] + records[1:]]
        for supplied in variants:
            with self.subTest(n=len(supplied)), self.assertRaises(ValueError):
                aggregate(supplied)

    def test_mixed_checkpoint_manifest_seed_or_wrong_sample_count_rejected(self):
        records = raw_fixture()
        for change in ({"checkpoint": "another"}, {"manifest_id": "heldout"}, {"training_seed": 9}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                aggregate([replace(records[0], **change)] + records[1:])
        with self.assertRaises(ValueError):
            aggregate(records, SAMPLES[:16])

    def test_cycle_fact_and_nested_binary_condition_contracts(self):
        valid, unparsed = raw_fixture()[1], raw_fixture()[0]
        for source, change in ((valid, {"has_cycle": None}), (unparsed, {"has_cycle": False}),
                               (valid, {"conditions": (True, False, True, False, False, False)}),
                               (valid, {"conditions": (1, 1, 1, 1, 1, 1)})):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(source, **change)


if __name__ == "__main__":
    unittest.main(verbosity=2)
