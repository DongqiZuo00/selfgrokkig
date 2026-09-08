"""Synthetic configuration checks; no model calls or experimental budget choices."""
from pathlib import Path
import unittest
from verge_control_protocol import segment_config, validate_control_config


class ControlProtocolTests(unittest.TestCase):
    def config(self, label="dense", round_index=0, phases=None, source=None):
        return segment_config(Path(__file__).resolve().parents[1], label, round_index,
            phases=phases or [{"kind":"target", "reward_mode":label, "tokens":524288}],
            starting_checkpoint=source or "checkpoints/verge_book_v2_initial_solver/resume_u0000",
            block_protocol="manifests/verge_control_v1_direct_protocol.json")

    def test_direct_modes_keep_fixed_contract_without_warm_start_metadata(self):
        for label in ("binary", "dense"):
            cfg = self.config(label)
            self.assertEqual(cfg["solver_rank"], 16)
            self.assertEqual(cfg["seed"], 42)
            self.assertEqual(cfg["proposal_count"], 0)
            self.assertIsNone(cfg["challenger_start"])
            self.assertFalse(cfg["final_evaluation_allowed"])
            self.assertNotIn("human-curriculum warm start", str(cfg["explicit_pilot_differences"]))

    def test_explicit_synthetic_boundary_is_preserved_not_inferred(self):
        phases = [{"kind":"target", "reward_mode":"dense", "tokens":8},
                  {"kind":"target", "reward_mode":"binary", "tokens":16}]
        cfg = self.config("dense_then_binary", phases=phases)
        self.assertEqual(cfg["train_tokens_per_branch"], 24)
        phases[0]["tokens"] = 999
        self.assertEqual(cfg["control_phases"][0]["tokens"], 8)

    def test_arbitrary_budget_or_reverse_schedule_rejected(self):
        for phases in ([{"kind":"target", "reward_mode":"dense", "tokens":524289}],
                       [{"kind":"target", "reward_mode":"dense", "tokens":0}],
                       [{"kind":"target", "reward_mode":"dense", "tokens":True}],
                       [{"kind":"curriculum", "reward_mode":"dense", "tokens":8}],
                       [{"kind":"target", "reward_mode":"binary", "tokens":8},
                        {"kind":"target", "reward_mode":"dense", "tokens":8}]):
            with self.assertRaises(ValueError):
                self.config("dense_then_binary", phases=phases)

    def test_foreign_or_noninitial_sources_are_rejected(self):
        for source in ("checkpoints/verge_book_v2_initial_solver_other/resume_u0000",
                       "checkpoints/verge_book_v2_initial_solver/resume_u0001",
                       "checkpoints/verge_book_v2_verge_r00_candidate_2/resume_u0031",
                       "checkpoints/../verge_book_v2_initial_solver/resume_u0000",
                       "/blue/checkpoints/verge_book_v2_initial_solver/resume_u0000"):
            with self.assertRaises(ValueError):
                self.config(source=source)

    def test_later_segment_requires_same_arm_immediate_predecessor(self):
        source = "checkpoints/verge_control_v1_dense_r00_direct/resume_u0030"
        self.assertEqual(self.config(round_index=1, source=source)["solver_start"], source)
        for invalid in (source.replace("dense", "binary"), source.replace("r00", "r01"),
                        "checkpoints/verge_book_v2_initial_solver/resume_u0000"):
            with self.assertRaises(ValueError):
                self.config(round_index=1, source=invalid)

    def test_primary_configs_and_foreign_namespaces_cannot_enter(self):
        cfg = self.config()
        for key, value in (("book_suite", "verge_book_v2"), ("book_atomic_round", True),
                           ("control_round", 1), ("control_only", False)):
            altered = dict(cfg, **{key:value})
            with self.assertRaises(ValueError):
                validate_control_config(altered, cfg["protocol_version"])
        with self.assertRaises(ValueError):
            validate_control_config(cfg, "verge_book_v2_verge_r00")

    def test_seed_sampler_optimizer_and_endpoint_changes_rejected(self):
        cfg = self.config()
        for key, value in (("seed", 43), ("weight_decay", .1), ("kl_beta", 0.),
                           ("solver_top_p", .9), ("completion_tokens", 4096),
                           ("maximum_updates_per_branch", 101), ("endpoint_instances", 2),
                           ("independent_prompt_streams", False), ("final_evaluation_allowed", True)):
            with self.assertRaises(ValueError):
                validate_control_config(dict(cfg, **{key:value}), cfg["protocol_version"])

    def test_rounds_and_protocol_reference_are_bounded(self):
        for label, r in (("qwen", 0), ("dense", 6), ("dense", -1), ("dense", True)):
            with self.assertRaises(ValueError):
                self.config(label, r)
        cfg = self.config()
        cfg["control_block_protocol"] = "../protocol.json"
        with self.assertRaises(ValueError):
            validate_control_config(cfg, cfg["protocol_version"])


if __name__ == "__main__":
    unittest.main()
