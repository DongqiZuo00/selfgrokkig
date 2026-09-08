"""Contract tests with synthetic fixtures; no model sampling or checkpoint hashes."""
import copy
import json
from pathlib import Path
import unittest
from verge_book_protocol import make_config, version, proposal_messages, arm_decision, PRIMARY_ARMS, SUITE
from verge_book_controller import schedule

EXP = Path(__file__).resolve().parents[1]


class BookContractTests(unittest.TestCase):
    def test_fresh_root_all_arms(self):
        configs = [make_config(EXP, arm, 0) for arm in PRIMARY_ARMS]
        self.assertEqual(len({c["solver_start"] for c in configs}), 1)
        self.assertEqual(len({c["challenger_start_checkpoint"] for c in configs}), 1)
        self.assertTrue(all(c["solver_rank"] == 16 and c["solver_learning_rate"] == 1e-5 for c in configs))
        self.assertTrue(all("v5_5" not in c["solver_start"] for c in configs))

    def test_v2_recovery_is_explicit_and_does_not_inherit_legacy_trajectory(self):
        cfg = make_config(EXP, "verge", 0)
        if SUITE == "verge_book_v2":
            self.assertTrue(cfg["independent_prompt_streams"])
            self.assertEqual(cfg["initial_root_copy_from_suite"], "verge_book_v1")
            self.assertEqual(cfg["random_stream_id"], "verge_book_v1_paired_r00")
            self.assertEqual(cfg["prior_book_versions"], [])
            prior = dict(cfg, book_suite="verge_book_v1")
            with self.assertRaises(ValueError):
                make_config(EXP, "verge", 1, {"config": prior})
        else:
            self.assertNotIn("independent_prompt_streams", cfg)

    def test_prior_pair_and_archive(self):
        first = make_config(EXP, "verge", 0)
        cfg = make_config(EXP, "verge", 1, {"config": first, "selected_solver": "solver_after",
                                            "selected_challenger": "teacher_after"})
        self.assertEqual(cfg["solver_start"], "solver_after")
        self.assertEqual(cfg["challenger_start_checkpoint"], "teacher_after")
        self.assertEqual(cfg["prior_book_versions"], [version("verge", 0)])
        with self.assertRaises(ValueError):
            make_config(EXP, "outcome", 1, {"config": first})

    def test_bounded_round_major_schedule(self):
        self.assertEqual(len(schedule()), 24)
        self.assertEqual(schedule()[:4], [(arm, 0) for arm in PRIMARY_ARMS])
        with self.assertRaises(ValueError):
            version("verge", 6)

    def test_paired_randomness_changes_across_rounds_not_arms(self):
        a, b = make_config(EXP, "verge", 0), make_config(EXP, "frozen", 0)
        self.assertEqual(a["random_stream_id"], b["random_stream_id"])
        c = make_config(EXP, "verge", 1, {"config": a, "selected_solver": "s", "selected_challenger": "q"})
        self.assertNotEqual(c["random_stream_id"], a["random_stream_id"])

    def test_no_human_history_in_initial_prompt(self):
        cfg = make_config(EXP, "verge", 0)
        initial = {"target": dict(counts=[0]*11, rates=[0]*11, rollouts=512,
                                  condition_names=list(range(11)), bottleneck_index=0)}
        prompt = json.dumps(proposal_messages(cfg, initial, []))
        self.assertNotIn("v5.5", prompt)
        self.assertNotIn("31, 0, 0", prompt)

    def fixture(self, mode):
        cfg = make_config(EXP, mode, 0)
        f = {"config": cfg, "initial": {"target": {"rates": [0]*11}},
             "generation": {"candidates": [{"spec": {"stages": [{}]}} for _ in range(3)]}}
        branches = {i: {"training": {"optimizer_steps": 1,
            "curriculum_stage_reward_counts": [{"draws": 8, "successes": n}]},
            "target": {"rates": [0]*10 + [.125 if i == 1 else 0]}} for i, n in enumerate([0,4,8],1)}
        d = {"challenger_rewards": [.2, .1, 0.], "selected": {"name": "candidate_1"},
             "evaluated_candidates": [{"eligible": True}]}
        return f, branches, d

    def test_reward_ablation_cannot_change_selector(self):
        for mode in PRIMARY_ARMS:
            f, branches, d = self.fixture(mode)
            result = arm_decision(f, branches, d)
            self.assertEqual(result["selected"], d["selected"])
            self.assertEqual(result["evaluated_candidates"], d["evaluated_candidates"])
        f, branches, d = self.fixture("uncertainty")
        self.assertEqual(arm_decision(f, branches, d)["challenger_rewards"], [0.,1.,0.])

    def test_frozen_teacher_does_not_update(self):
        f, branches, d = self.fixture("frozen")
        self.assertFalse(arm_decision(f, branches, d)["challenger_update_allowed"])

    def test_outcome_stationary_noise_is_not_credit(self):
        f, branches, d = self.fixture("outcome")
        branches[1]["training"]["optimizer_steps"] = 0
        self.assertEqual(arm_decision(f, branches, d)["challenger_rewards"], [0.,0.,0.])


if __name__ == "__main__":
    unittest.main()
