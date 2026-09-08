"""Synthetic CPU contract tests, not formal benchmark grouping or model results."""
import copy
import unittest

from operational_stages import (
    HINT_START, HintProbe, RegisteredFamily, RejectionSamplingBuffer,
    VerifiedProgramLibrary, evaluate_transform, generate_cases,
    instance_tuple_from_manifest, render_target_prompt, validate_distribution, validate_stage,
)


DIST = {"alphabet": "ab", "min_length": 0, "max_length": 4, "corner_cases": ["", "ab", "b"]}
REGISTRY = {"reverse": RegisteredFamily(lambda tape, spec: tape[::-1], "synthetic-reverse-tests-v1")}
TARGET_DEF = {"family": "reverse", "tier": 0, "alphabet": "ab"}
PROTECTED = [instance_tuple_from_manifest(TARGET_DEF, suite) for suite in (["aaaa"], ["bbbb"], ["aabb"])]
TRAIN_TUPLE = instance_tuple_from_manifest(TARGET_DEF, ["", "ab", "b"])
HINT_TUPLE = instance_tuple_from_manifest(TARGET_DEF, ["a", "b"])


def stage(kind="registered"):
    base = {"kind": kind, "distribution": copy.deepcopy(DIST), "instance_tuples": [list(TRAIN_TUPLE)]}
    if kind in ("tape_transform", "restricted_transform"):
        base["expression"] = {"op": "reverse", "arg": {"op": "input"}}
    else:
        base.update(family="reverse", tier=0, mutation_tier=0)
    if kind == "hinted_target":
        base.update(target_size=4, hint_id="hint-1", hint_relation="smaller_train")
    return base


def library(**overrides):
    lib = VerifiedProgramLibrary()
    kwargs = dict(entry_id="hint-1", program="SYNTHETIC verified reverse example",
                  family="reverse", instance_size=2, source_instance_tuple=HINT_TUPLE,
                  source_split="train", verification_id="fixture-verifier-log-1",
                  verifier_protocol_id="synthetic-reverse-tests-v1",
                  verification_test_ids=["fixture-a", "fixture-b"], exact_pass=[1, 1])
    kwargs.update(overrides)
    lib.add(**kwargs)
    return lib


def probe(with_count=4, without_count=0):
    return HintProbe(with_hint=[[1] * with_count + [0] * (8 - with_count) for _ in range(32)],
                     without_hint=[[1] * without_count + [0] * (8 - without_count) for _ in range(32)],
                     instance_ids=[f"fixture-train-{i}" for i in range(32)],
                     rollout_seed_ids=[list(range(i * 8, i * 8 + 8)) for i in range(32)],
                     condition_id="synthetic-binary-full-pass-v1")


def candidate(name="g1", gamma=0.2, ci=(0.1, 0.3)):
    return {"branch_id": name, "gamma": gamma, "paired_ci": list(ci),
            "curriculum": [stage("tape_transform")], "evidence_id": f"fixture-{name}-rate-log"}


def add_round(buffer, round_id="round-1", seed=1, candidates=None, j_plus=None):
    candidates = [candidate()] if candidates is None else candidates
    j_plus = ["g1"] if j_plus is None else j_plus
    return buffer.add_round(round_id=round_id, seed=seed,
                            context={"synthetic_fixture": True, "delta": [[0.1, 0.1]]},
                            base={"checkpoint_id": "synthetic-base", "cell": [1, 0]},
                            candidates=candidates, j_plus=j_plus,
                            provenance={"run_id": "fixture-run", "condition_id": "fixed-kappa-1",
                                        "evidence_id": f"fixture-{seed}-{round_id}"})


class StageTests(unittest.TestCase):
    def validate(self, spec, **kwargs):
        return validate_stage(spec, registry=REGISTRY, protected_tuples=PROTECTED, **kwargs)

    def test_three_stage_sources_have_exactly_four_checks(self):
        for kind in ("registered", "tape_transform", "hinted_target"):
            kwargs = {"library": library(), "hint_probe": probe()} if kind == "hinted_target" else {}
            result = self.validate(stage(kind), **kwargs)
            self.assertTrue(result.accepted, result.details)
            self.assertEqual(set(result.checks), {"executable", "nonconstant", "no_leakage", "hint_regret"})
            self.assertEqual(result.checks["hint_regret"], True if kind == "hinted_target" else "NA")
        alias = self.validate(stage("restricted_transform"))
        self.assertTrue(alias.accepted)
        self.assertEqual(alias.details["canonical_kind"], "tape_transform")

    def test_distribution_is_explicit_and_frozen_generator_repeats(self):
        self.assertEqual(generate_cases(DIST, seed=4), generate_cases(DIST, seed=4))
        for update in ({"alphabet": "aa"}, {"max_length": 1025}, {"min_length": 5},
                       {"corner_cases": ["c", "ab"]}, {"corner_cases": ["a", "a"]},
                       {"min_length": True}, {"surprise": "x"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_distribution({**DIST, **update})

    def test_dsl_operations_are_data_only_and_bounded(self):
        inp = {"op": "input"}
        self.assertEqual(evaluate_transform({"op": "rotate", "arg": inp, "offset": 1}, "ab"), "ba")
        self.assertEqual(evaluate_transform({"op": "rotate", "arg": inp, "offset": 1}, ""), "")
        self.assertEqual(evaluate_transform({"op": "slice", "arg": inp, "start": 0, "stop": 1}, "ab"), "a")
        self.assertEqual(evaluate_transform({"op": "replace", "arg": inp, "old": "a", "new": "b"}, "ab"), "bb")
        self.assertTrue(evaluate_transform({"op": "contains", "arg": inp, "value": "b"}, "ab"))
        self.assertTrue(evaluate_transform({"op": "equals", "left": inp, "right": {"op": "literal", "value": "ab"}}, "ab"))
        for expr in ({"op": "exec", "code": "open('bad','w')"},
                     {"op": "input", "__class__": "bad"},
                     {"op": "concat", "args": [{"op": "literal", "value": "a" * 2000}] * 4}):
            with self.assertRaises(ValueError):
                evaluate_transform(expr, "ab")
        nested = inp
        for _ in range(10):
            nested = {"op": "reverse", "arg": nested}
        with self.assertRaises(ValueError):
            evaluate_transform(nested, "ab")

    def test_nonconstant_is_checked_on_corners_not_random_samples(self):
        spec = stage("restricted_transform")
        spec["expression"] = {"op": "literal", "value": "constant"}
        result = self.validate(spec)
        self.assertFalse(result.accepted)
        self.assertTrue(result.checks["executable"])
        self.assertFalse(result.checks["nonconstant"])
        spec["expression"] = {"op": "contains", "arg": {"op": "input"}, "value": "b"}
        self.assertTrue(self.validate(spec).accepted)

    def test_exact_tuple_leakage_all_three_protected_splits(self):
        for params in PROTECTED:
            spec = stage()
            spec["instance_tuples"] = [list(params)]
            self.assertFalse(self.validate(spec).checks["no_leakage"])
        self.assertFalse(validate_stage(stage(), registry=REGISTRY).accepted)
        spec = stage()
        spec["instance_tuples"] = []
        self.assertFalse(self.validate(spec).accepted)
        same_inputs_reordered = instance_tuple_from_manifest(TARGET_DEF, ["b", "ab", ""])
        self.assertEqual(same_inputs_reordered, TRAIN_TUPLE)
        spec["instance_tuples"] = [["train", *TRAIN_TUPLE]]
        self.assertFalse(self.validate(spec).accepted)
        with self.assertRaises(ValueError):
            instance_tuple_from_manifest({**TARGET_DEF, "split": "test"}, ["a"])

    def test_registered_family_tier_and_output_contract(self):
        spec = stage()
        spec["tier"] = 9
        self.assertFalse(self.validate(spec).accepted)
        broken = {"reverse": RegisteredFamily(lambda tape, spec: 3 / 0, "fixture-broken")}
        result = validate_stage(stage(), registry=broken, protected_tuples=PROTECTED)
        self.assertFalse(result.checks["executable"])

    def test_hint_probe_512_paired_budget_and_two_conditions(self):
        result = self.validate(stage("hinted_target"), library=library(), hint_probe=probe())
        details = result.details["hint_probe"]
        self.assertEqual(details["total_rollouts"], 512)
        self.assertEqual(details["p_with_hint"], 0.5)
        self.assertEqual(details["p_without_hint"], 0.0)
        for p in (probe(0, 0), probe(3, 4), probe(8, 8)):
            self.assertFalse(self.validate(stage("hinted_target"), library=library(), hint_probe=p).accepted)
        almost_saturated = probe(8, 7)
        # Raise no-hint success from 224/256 to 231/256 (>0.9).
        for i in range(7):
            almost_saturated.without_hint[i][-1] = 1
        self.assertFalse(self.validate(stage("hinted_target"), library=library(), hint_probe=almost_saturated).accepted)

    def test_mixed_reward_group_is_diagnostic_not_fifth_gate(self):
        result = self.validate(stage("hinted_target"), library=library(), hint_probe=probe(8, 0))
        self.assertTrue(result.accepted)
        self.assertEqual(result.details["hint_probe"]["mixed_groups_with_hint"], 0)
        self.assertEqual(len(result.checks), 4)
        self.assertFalse(self.validate(stage(), hint_probe=probe()).accepted)

    def test_hint_probe_rejects_wrong_shape_nonbinary_and_unpaired_ids(self):
        for field, value in (("with_hint", [[0] * 8 for _ in range(31)]),
                             ("without_hint", [[0.5] * 8 for _ in range(32)]),
                             ("instance_ids", ["duplicate"] * 32),
                             ("rollout_seed_ids", [[0] * 8 for _ in range(32)]),
                             ("with_hint", [[[1] * 8 for _ in range(32)]] * 2)):
            p = probe()
            kwargs = dict(vars(p))
            kwargs[field] = value
            self.assertFalse(self.validate(stage("hinted_target"), library=library(), hint_probe=HintProbe(**kwargs)).accepted)

    def test_library_requires_verified_train_provenance(self):
        for update in ({"exact_pass": [1, 0]}, {"exact_pass": []}, {"source_split": "test"},
                       {"verification_id": ""}, {"verification_test_ids": ["same", "same"]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                library(**update)
        item = library().entries["hint-1"]
        self.assertEqual(len(item.program_sha256), 64)
        self.assertEqual(item.verifier_protocol_id, "synthetic-reverse-tests-v1")

    def test_hint_smaller_relationship_and_library_source_leakage(self):
        self.assertFalse(self.validate(stage("hinted_target"), library=library(instance_size=4), hint_probe=probe()).accepted)
        self.assertFalse(self.validate(stage("hinted_target"), library=library(family="different"), hint_probe=probe()).accepted)
        result = self.validate(stage("hinted_target"), library=library(source_instance_tuple=PROTECTED[0]), hint_probe=probe())
        self.assertFalse(result.checks["no_leakage"])
        self.assertFalse(self.validate(stage("hinted_target"), library=library()).accepted)

    def test_final_target_prompt_rejects_hint_contamination(self):
        hint = library().entries["hint-1"]
        text = render_target_prompt("Solve frozen target.", hint=hint)
        self.assertIn(HINT_START, text)
        self.assertEqual(render_target_prompt("Solve frozen target.", final_evaluation=True), "Solve frozen target.")
        with self.assertRaises(ValueError):
            render_target_prompt("Solve frozen target.", hint=hint, final_evaluation=True)
        with self.assertRaises(ValueError):
            render_target_prompt(text, final_evaluation=True)


class BufferTests(unittest.TestCase):
    def test_empty_jplus_records_archive_without_update(self):
        buf = RejectionSamplingBuffer(1)
        add_round(buf, j_plus=[])
        self.assertEqual(len(buf.archive), 1)
        self.assertEqual(len(buf.examples), 0)
        self.assertEqual(buf.new_count, 0)
        self.assertIsNone(buf.plan_update())

    def test_jplus_uses_positive_significant_gain_only_atomic_insert(self):
        for bad in (candidate(gamma=0), candidate(gamma=-0.1), candidate(ci=(-0.1, 0.3)), candidate(ci=(0, 0.3))):
            buf = RejectionSamplingBuffer(1)
            with self.assertRaises(ValueError):
                add_round(buf, candidates=[bad])
            self.assertEqual(buf.archive, [])
            self.assertEqual(buf.examples, [])
        buf = RejectionSamplingBuffer(1)
        add_round(buf, candidates=[candidate(), candidate("g2", gamma=-0.1)], j_plus=["g1"])
        self.assertEqual([x["branch_id"] for x in buf.examples], ["g1"])
        self.assertEqual(len(buf.archive[0]["candidates"]), 2)

    def test_cross_seed_cumulative_data_threshold_and_single_epoch(self):
        buf = RejectionSamplingBuffer(2)
        add_round(buf, seed=1)
        self.assertIsNone(buf.plan_update())
        add_round(buf, seed=2)
        plan = buf.plan_update()
        self.assertEqual(plan["epochs"], 1)
        self.assertEqual(plan["new_count_at_plan"], 2)
        self.assertEqual(len(plan["cumulative_examples"]), 2)
        self.assertEqual({x["seed"] for x in plan["cumulative_examples"]}, {1, 2})
        self.assertIsNone(buf.plan_update())
        buf.complete_update(plan["plan_id"], success=True, checkpoint_id="fixture-q1", training_log_id="fixture-training1")
        self.assertEqual(buf.new_count, 0)
        add_round(buf, round_id="round-2", seed=1)
        add_round(buf, round_id="round-2", seed=2)
        self.assertEqual(len(buf.plan_update()["cumulative_examples"]), 4)

    def test_failed_training_keeps_new_examples_and_retry_cumulative(self):
        buf = RejectionSamplingBuffer(1)
        add_round(buf)
        plan = buf.plan_update()
        buf.complete_update(plan["plan_id"], success=False, failure_reason="fixture simulated trainer failure")
        self.assertEqual(buf.new_count, 1)
        self.assertEqual(buf.trained_ids, set())
        retry = buf.plan_update()
        self.assertEqual(retry["dataset_sha256"], plan["dataset_sha256"])
        self.assertNotEqual(retry["plan_id"], plan["plan_id"])
        self.assertEqual(len(buf.failed_updates), 1)

    def test_examples_arriving_during_training_remain_new(self):
        buf = RejectionSamplingBuffer(1)
        add_round(buf)
        plan = buf.plan_update()
        add_round(buf, seed=2)
        with self.assertRaises(ValueError):
            buf.complete_update(plan["plan_id"], success=True)
        self.assertEqual(buf.new_count, 2)
        buf.complete_update(plan["plan_id"], success=True, checkpoint_id="fixture-q", training_log_id="fixture-log")
        self.assertEqual(buf.new_count, 1)
        self.assertEqual(len(buf.plan_update()["cumulative_examples"]), 2)

    def test_provenance_is_snapshotted_and_duplicate_round_rejected(self):
        buf = RejectionSamplingBuffer(1)
        c = candidate()
        add_round(buf, candidates=[c])
        c["curriculum"].clear()
        self.assertEqual(len(buf.examples[0]["curriculum"]), 1)
        with self.assertRaises(ValueError):
            add_round(buf)
        self.assertEqual(len(buf.archive), 1)
        snapshot = buf.to_dict()
        self.assertEqual(snapshot["examples"][0]["provenance"]["evidence_id"], "fixture-1-round-1")
        self.assertEqual(len(snapshot["examples"][0]["example_id"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
