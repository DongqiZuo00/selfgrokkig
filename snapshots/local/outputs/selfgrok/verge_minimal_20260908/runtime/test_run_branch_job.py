"""Readiness/dispatch fixture tests; injected executor never loads a model."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "round/fixtures")]
from fixture_round import request as fixture_request
from test_execute_round import FixtureSolver
from execute_round import execute_base
from round_cli import build_plan, canonical, sha, write_json
import run_branch_job as wrapper


class BranchWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prep = self.root / "round_prepare_fixture"
        self.prep.mkdir()
        adapter = self.root / "source_base"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_text("0")
        req = fixture_request()
        req["base_checkpoint"] = str(adapter)
        req["proposal"]["base_checkpoint"] = str(adapter)
        req["challenger"]["raw_output"] = canonical(req["proposal"])
        req["challenger"]["evidence_id"] = str(self.prep / "challenger_sample_0.json")
        req["selection"]["prompt_view_sha256"] = sha("fixture-prompt")
        self.plan = build_plan(req, _fixture=True)
        write_json(self.prep / "challenger_sample_0.json", {"mode": "unit_fixture",
            "checkpoint": req["challenger"]["checkpoint"], "raw_output": req["challenger"]["raw_output"]})
        cases = [{"test_id": t, "input": t, "expected_output": t, "expected_accepted": True,
                  "check_output": True} for t in ("t1", "t2", "t3", "t4")]
        messages = [{"role": "user", "content": "explicit unit fixture"}]
        rows = {"manifest_id": self.plan["selection"]["manifest_id"],
            "prompt_view_sha256": self.plan["selection"]["prompt_view_sha256"],
            "train": [{"id": "train-1", "messages": messages, "ground_truth": cases}],
            "selection": [{"id": i, "messages": messages, "ground_truth": cases} for i in self.plan["selection"]["instance_ids"]]}
        self.target_path = self.root / "target_rows.json"
        self.catalogue_path = self.root / "catalogue.json"
        write_json(self.target_path, rows)
        write_json(self.catalogue_path, {"stages": [{**s, "rows": rows["train"]} for s in req["proposal"]["curricula"]["g1"]]})
        execute_base(self.plan, rows, self.prep / "base", solver_factory=FixtureSolver,
                     fingerprint=lambda solver: sha(solver.value), _fixture=True)
        write_json(self.prep / "plan.json", self.plan)
        self.ready = {"status": "real_base_and_challenger_complete", "plan": str(self.prep / "plan.json"),
            "base_fragment": str(self.prep / "base/fragment.json"), "plan_sha256": self.plan["plan_sha256"],
            "curricula": {k: [s["stage_id"] for s in v] for k, v in req["proposal"]["curricula"].items()}}
        write_json(self.prep / "READY.json", self.ready)

    def test_current_prepare_ready_without_future_optional_hashes_is_compatible(self):
        resolved = wrapper.resolve_ready(self.prep, _fixture=True)
        self.assertEqual(resolved["plan"]["plan_sha256"], self.plan["plan_sha256"])
        calls = []
        def fake_executor(plan, rows, stage_map, base, branch, output, **kwargs):
            calls.append((branch, len(rows["selection"]), kwargs["_fixture"]))
            return {"fragment_role": branch, "execution": {"explicit_unit_fixture": True}}
        result = wrapper.run_job(self.root / "job", plan_root=self.prep, branch_id="g1",
            target_rows=self.target_path, catalogue=self.catalogue_path, _fixture=True, executor=fake_executor)
        self.assertEqual(result["fragment_role"], "g1")
        self.assertEqual(calls, [("g1", 8, True)])
        launch = wrapper.load_json(self.root / "job/LAUNCH.json")
        self.assertEqual(launch["catalogue_sha256"], wrapper.file_sha(self.catalogue_path))
        self.assertTrue((self.root / "job/COMPLETE.json").exists())

    def test_ready_requires_completed_actual_preparation_and_matching_q(self):
        path = self.prep / "READY.json"
        changed = dict(self.ready, status="preparing")
        path.write_text(canonical(changed))
        with self.assertRaisesRegex(ValueError, "not READY"):
            wrapper.resolve_ready(self.prep, _fixture=True)
        path.write_text(canonical(self.ready))
        source = self.prep / "challenger_sample_0.json"
        evidence = wrapper.load_json(source)
        evidence["raw_output"] = "{}"
        source.write_text(canonical(evidence))
        with self.assertRaisesRegex(ValueError, "Challenger generation"):
            wrapper.resolve_ready(self.prep, _fixture=True)

    def test_explicit_environment_selects_one_branch_and_failure_preserves_launch(self):
        def failing(*args, **kwargs):
            raise RuntimeError("intentional fixture failure")
        with patch.dict(wrapper.os.environ, {"VERGE_PLAN_ROOT": str(self.prep), "VERGE_BRANCH": "direct"}):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                wrapper.run_job(self.root / "failed_job", target_rows=self.target_path, catalogue=self.catalogue_path,
                                _fixture=True, executor=failing)
        self.assertTrue((self.root / "failed_job/LAUNCH.json").exists())
        self.assertTrue(wrapper.load_json(self.root / "failed_job/FAILED.json")["partial_evidence_preserved"])

    def test_optional_frozen_catalogue_hash_rejects_changed_wording_before_executor(self):
        self.ready["catalogue_sha256"] = "0" * 64
        (self.prep / "READY.json").write_text(canonical(self.ready))
        with self.assertRaisesRegex(ValueError, "catalogue wording"):
            wrapper.run_job(self.root / "never_created", plan_root=self.prep, branch_id="g1",
                target_rows=self.target_path, catalogue=self.catalogue_path, _fixture=True,
                executor=lambda *a, **kw: self.fail("executor should not be reached"))
        self.assertFalse((self.root / "never_created").exists())

    def test_frozen_runtime_contract_checks_every_file_and_is_copied_to_job(self):
        contract = wrapper.load_json(wrapper.ROOT / "configs/round1_runtime_contract.json")
        contract["prepare_root"] = str(self.prep)
        contract["training_generated_token_budget"] = {"per_branch": self.plan["B"], "four_branch_total": 4 * self.plan["B"]}
        path = self.root / "runtime_contract.json"
        write_json(path, contract)
        validated, report = wrapper.verify_runtime_contract(path, self.prep, self.plan)
        self.assertEqual(len(report["verified_files"]), len(contract["files"]))
        wrapper.run_job(self.root / "contract_job", plan_root=self.prep, branch_id="direct",
            target_rows=self.target_path, catalogue=self.catalogue_path, runtime_contract=path, _fixture=True,
            executor=lambda *args, **kwargs: {"fragment_role": "direct", "execution": {"explicit_unit_fixture": True}})
        self.assertEqual(wrapper.load_json(self.root / "contract_job/runtime_contract.json"), contract)
        contract["files"]["runtime/hf_backend.py"] = "0" * 64
        path.write_text(canonical(contract))
        with self.assertRaisesRegex(ValueError, "frozen runtime file changed"):
            wrapper.verify_runtime_contract(path, self.prep, self.plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
