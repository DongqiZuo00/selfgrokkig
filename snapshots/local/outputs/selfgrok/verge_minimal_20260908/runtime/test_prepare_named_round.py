"""CPU contracts only. Invented raw records stay in temporary fixture directories."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_named_round as p


class NamedPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = p.load_view(p.VIEW)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="named_unit_fixture_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def screen_fixture(self, *, success=True):
        screen = self.root / "screen_fixture"
        screen.mkdir()
        adapter = self.root / "SYNTHETIC_UNIT_ADAPTER"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_text("SYNTHETIC UNIT FIXTURE")
        tasks = p.read(p.ROOT / "benchmarks/generated/naming_prompt_v2/screen_tasks.json")
        task_path = self.root / "fixture_tasks.json"
        p.write_json(task_path, tasks)
        p.write_json(screen / "frozen_screen.json", {"task_file": str(task_path), "task_file_sha256": p.sha256(task_path),
            "screen": tasks, "original_adapter_sha256": p.sha256(adapter / "adapter_model.safetensors")})
        summaries = []
        for index, task in enumerate(tasks["tasks"]):
            rewards = [0, 1] * 4 if success and index == 0 else [0] * 8
            tests = task["row"]["ground_truth"]
            raw = {"mode": "SYNTHETIC_UNIT_FIXTURE", "adapter": str(adapter.resolve()), "optimizer_steps_before": 0,
                "instance_id": task["row"]["id"], "seed": task["seed"], "max_new_tokens": task["cap"],
                "n": 8, "decoding_policy": "dsl_grammar_v1", "rewards": rewards,
                "prompt": task["row"]["messages"][0]["content"], "completion_token_ids": [[7]] * 8,
                "verifier_completions": ["SYNTHETIC_UNIT_PROGRAM"] * 8,
                "verification": [{"reward": reward, "parse_valid": True,
                    "per_test": [{"test_id": t.get("test_id", str(j)), "input": t["input"], "pass": reward}
                                 for j, t in enumerate(tests)]} for reward in rewards],
                "generated_tokens": 8, "binary_successes": sum(rewards), "mixed_group": len(set(rewards)) == 2,
                "parse_valid": 8}
            raw_path = screen / f"raw_fixture_{index}.json"
            p.write_json(raw_path, raw)
            summaries.append({"stage_id": task["stage_id"], "variant": task["variant"],
                "raw_path": str(raw_path), **{k: raw[k] for k in ("binary_successes", "mixed_group", "parse_valid", "generated_tokens")}})
        p.write_json(screen / "SUMMARY.json", {"status": "completed", "mode": "SYNTHETIC_UNIT_FIXTURE",
            "source_adapter_unchanged": True, "rollouts": 96, "summaries": summaries,
            "mixed_groups": int(success), "binary_successes": 4 if success else 0, "actual_optimizer_steps": int(success)})
        return screen, adapter

    def history_fixture(self):
        sys.path.insert(0, str(p.ROOT / "round/fixtures"))
        from fixture_round import request, exchange
        from round_cli import score_exchange
        req = request()
        req["selection"]["prompt_view_sha256"] = p.sha("SYNTHETIC_UNIT_OLD_VIEW")
        plan = p.build_plan(req, _fixture=True)
        old = self.root / "history_fixture"
        old.mkdir()
        raw = exchange(old, plan)
        for evaluation in raw["evaluations"].values():
            for record in evaluation["records"]:
                record["prompt_view_sha256"] = req["selection"]["prompt_view_sha256"]
        scored = score_exchange(plan, raw, _fixture=True)
        for name in p.HISTORY_FILES:
            p.write_json(old / name, raw if name == "exchange.json" else scored[name[:-5]])
        planpath = old / "old_plan.json"
        p.write_json(planpath, plan)
        return old, planpath

    def test_real_frozen_v2_input_loader_preserves_all_tests_and_four_gates(self):
        self.assertEqual(len(self.loaded["rows"]["train"]), 128)
        self.assertEqual(len(self.loaded["rows"]["selection"]), 8)
        self.assertEqual(len(self.loaded["stages"]), 9)
        self.assertEqual(self.loaded["baseline"]["evaluation_requests"][0]["n"], 32)
        for validation in self.loaded["validations"].values():
            self.assertEqual(set(validation["checks"]), {"executable", "nonconstant", "no_leakage", "hint_regret"})
            self.assertEqual(validation["checks"]["hint_regret"], "NA")

    def test_schema_is_identical_to_old_menu_and_has_no_hint_choice(self):
        import prepare_round
        self.assertEqual(p.proposal_schema(self.loaded["stages"]), prepare_round.proposal_schema(self.loaded["stages"]))
        self.assertEqual({s["kind"] for s in self.loaded["stages"]}, {"registered", "tape_transform"})

    def test_dry_run_reports_missing_history_screen_without_model_or_ready(self):
        with patch.object(p, "execute_base", side_effect=AssertionError("must not load model")):
            result = p.main(self.root / "dry", screen=self.root / "missing_screen",
                            history=self.root / "missing_history", history_plan=self.root / "missing_plan", dry_run=True)
        self.assertEqual({d["dependency"] for d in result["pending_dependencies"]}, {"history", "screen"})
        self.assertFalse(result["models_loaded"])
        self.assertFalse((self.root / "dry/READY.json").exists())
        self.assertTrue((self.root / "dry/challenger_schema.json").exists())

    def test_verified_program_index_contains_only_actual_success_slots(self):
        screen, adapter = self.screen_fixture()
        checked = p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter, _fixture=True)
        self.assertEqual([r["rollout_slot"] for r in checked["library"]], [1, 3, 5, 7])
        self.assertTrue(all(not r["hint_regret_validated"] for r in checked["library"]))
        self.assertTrue(all(r["evidence_file_sha256"] for r in checked["library"]))
        self.assertFalse(checked["run_preparation_diagnostic"]["is_stage_validator_gate"])

    def test_zero_mixed_is_a_resource_diagnostic_not_a_fifth_gate(self):
        screen, adapter = self.screen_fixture(success=False)
        checked = p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter, _fixture=True)
        self.assertEqual(checked["library"], [])
        self.assertEqual(checked["run_preparation_diagnostic"]["mixed_groups"], 0)

    def test_fixture_cannot_be_promoted_to_real_screen(self):
        screen, adapter = self.screen_fixture()
        with self.assertRaisesRegex(ValueError, "completed v2"):
            p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter)

    def test_screen_old_prompt_and_forged_reward_are_rejected(self):
        screen, adapter = self.screen_fixture()
        rawpath = screen / "raw_fixture_0.json"
        raw = p.read(rawpath)
        changed = copy.deepcopy(raw)
        changed["prompt"] = "OLD PROMPT"
        p.write_json(rawpath, changed)
        with self.assertRaisesRegex(ValueError, "new naming view"):
            p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter, _fixture=True)
        raw["verification"][1]["per_test"][0]["pass"] = 0
        p.write_json(rawpath, raw)
        with self.assertRaisesRegex(ValueError, "full exact"):
            p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter, _fixture=True)

    def test_history_is_read_verbatim_and_never_current_incumbent(self):
        old, planpath = self.history_fixture()
        history = p.load_history(old, planpath, self.loaded["rows"]["prompt_view_sha256"], _fixture=True)
        self.assertEqual(history["per_stage_delta"], p.read(old / "delta.json"))
        self.assertEqual(history["round_score"], p.read(old / "round_score.json"))
        self.assertFalse(history["eligible_as_current_incumbents"])
        with self.assertRaisesRegex(ValueError, "distinct old prompt"):
            p.load_history(old, planpath, history["prompt_view_sha256"], _fixture=True)
        with self.assertRaisesRegex(ValueError, "actual model"):
            p.load_history(old, planpath, self.loaded["rows"]["prompt_view_sha256"])

    def test_history_exchange_and_delta_tampering_rejected(self):
        old, planpath = self.history_fixture()
        delta = p.read(old / "delta.json")
        delta["g1"]["gamma"] = -123
        p.write_json(old / "delta.json", delta)
        with self.assertRaisesRegex(ValueError, "Delta and Gamma"):
            p.load_history(old, planpath, self.loaded["rows"]["prompt_view_sha256"], _fixture=True)
        raw = p.read(old / "exchange.json")
        raw["tampered"] = True
        p.write_json(old / "exchange.json", raw)
        with self.assertRaisesRegex(ValueError, "changed after scoring"):
            p.load_history(old, planpath, self.loaded["rows"]["prompt_view_sha256"], _fixture=True)

    def test_new_context_rho_comes_only_from_new_256_records(self):
        old, planpath = self.history_fixture()
        history = p.load_history(old, planpath, self.loaded["rows"]["prompt_view_sha256"], _fixture=True)
        screen, adapter = self.screen_fixture()
        screen_data = p.load_screen(screen, self.loaded["catalogue"], initial_solver=adapter, _fixture=True)
        selection = self.loaded["baseline"]["selection"]
        fragment = {"baseline_contract_sha256": self.loaded["baseline"]["baseline_contract_sha256"],
                    "evaluations": {"base": {"records": [
                        {"checkpoint": str(p.INITIAL_SOLVER), "training_seed": p.SEED,
                         "manifest_id": selection["manifest_id"], "instance_id": instance, "sample_id": sample,
                         "conditions": [True, False, False, False, False, False], "has_cycle": False,
                         "prompt_view_sha256": self.loaded["rows"]["prompt_view_sha256"], "hint": None}
                        for instance in selection["instance_ids"] for sample in selection["sample_ids"]]}}}
        context = p.measured_context(fragment, self.loaded, {"history": history, "screen": screen_data})
        self.assertEqual(len(context["incumbents"]), 1)
        self.assertEqual(context["incumbents"][0]["rho"], [1, 0, 0, 0, 0, 0])
        self.assertEqual(context["archive"], [])
        self.assertNotEqual(context["current_prompt_view_sha256"], context["old_prompt_view_history"]["prompt_view_sha256"])
        self.assertEqual(context["training"]["B"], 65536)
        fragment["evaluations"]["base"]["records"][0]["prompt_view_sha256"] = history["prompt_view_sha256"]
        with self.assertRaisesRegex(ValueError, "new unhinted view"):
            p.measured_context(fragment, self.loaded, {"history": history, "screen": screen_data})


if __name__ == "__main__":
    unittest.main(verbosity=2)
