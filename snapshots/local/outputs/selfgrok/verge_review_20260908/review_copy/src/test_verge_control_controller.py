"""Synthetic Slurm dependency and recovery tests: never submit real jobs."""
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_control_controller as ctl
import test_verge_control_block as block_tests


class ControlControllerTests(unittest.TestCase):
    def fixture(self, exp):
        helper = block_tests.ControlBlockTests()
        helper.fixture(exp)
        helper.write(exp / "raw_results/verge_book_v2/active_wave.json",{"finish":"999"})
        return helper

    def patches(self, exp):
        stack = ExitStack()
        stack.enter_context(patch.object(ctl,"EXP",exp))
        stack.enter_context(patch.object(ctl,"ROOT",exp / "raw_results/verge_control_v1"))
        stack.enter_context(patch.object(ctl,"dispatch_lock",lambda:nullcontext()))
        stack.enter_context(patch.dict(ctl.os.environ,{"VERGE_BOOK_SUITE":"verge_book_v2","VERGE_CONTROL_SUITE":"verge_control_v1"}))
        return stack

    def test_no_submit_without_primary_completion_or_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            with self.patches(exp), patch.object(ctl.subprocess,"run") as run:
                with self.assertRaises(RuntimeError):
                    ctl.dispatch()
                run.assert_not_called()
                self.fixture(exp)
                (exp / ctl.PLAN).unlink()
                with self.assertRaises(RuntimeError):
                    ctl.dispatch()
                run.assert_not_called()

    def test_first_wave_is_bounded_dependent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            self.fixture(exp)
            with self.patches(exp), patch.object(ctl.subprocess,"run",side_effect=[
                    SimpleNamespace(stdout="1000\n"),SimpleNamespace(stdout="1001\n")]) as run:
                result = ctl.dispatch()
                self.assertEqual(result["workers"],"1000")
                self.assertEqual(result["coordinator"],"1001")
                self.assertIn("--array=0-2%2",run.call_args_list[0].args[0])
                self.assertIn("--dependency=afterok:999",run.call_args_list[0].args[0])
                self.assertIn("--dependency=afterok:1000",run.call_args_list[1].args[0])
                self.assertEqual(ctl.dispatch(),result)
                self.assertEqual(run.call_count,2)
                configs = [json.loads((exp / "manifests" / (ctl.version(label,0)+".json")).read_text()) for label in ctl.LABELS]
                self.assertEqual(len({c["solver_start"] for c in configs}),1)
                self.assertTrue(all(c["train_tokens_per_branch"] == 524288 for c in configs))

    def test_uncertain_sbatch_response_cannot_duplicate_gpu_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            self.fixture(exp)
            with self.patches(exp), patch.object(ctl.subprocess,"run",return_value=SimpleNamespace(stdout="")) as run:
                with self.assertRaises(RuntimeError):
                    ctl.dispatch()
                with self.assertRaises(RuntimeError):
                    ctl.dispatch()
                self.assertEqual(run.call_count,1)

    def test_confirmed_intent_recovers_missing_outer_job_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            self.fixture(exp)
            with self.patches(exp), patch.object(ctl.subprocess,"run",side_effect=[
                    SimpleNamespace(stdout="1000"),SimpleNamespace(stdout="1001")]) as run:
                first = ctl.dispatch()
                (ctl.ROOT / "jobs/round_00.json").unlink()
                self.assertEqual(ctl.dispatch(),first)
                self.assertEqual(run.call_count,2)

    def test_changed_frozen_plan_blocks_existing_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = Path(temp)
            helper = self.fixture(exp)
            plan = helper.block()
            plan["dense_warmup_loss_tokens"] = 24
            helper.write(exp / ctl.PLAN,plan)
            with self.patches(exp), patch.object(ctl.subprocess,"run") as run:
                with self.assertRaises(RuntimeError):
                    ctl.dispatch()
                run.assert_not_called()

    def test_incomplete_wave_never_advances(self):
        with patch.object(ctl,"completed",return_value=None), patch.object(ctl,"dispatch") as dispatch:
            with self.assertRaises(RuntimeError):
                ctl.coordinate(0)
            dispatch.assert_not_called()

    def test_coordinate_deep_checks_all_three_arms(self):
        with patch.object(ctl,"completed",return_value={"valid":True}) as completed, patch.object(ctl,"dispatch",return_value={"round":1}) as dispatch:
            self.assertEqual(ctl.coordinate(0),{"round":1})
            self.assertEqual(completed.call_count,3)
            self.assertTrue(all(c.kwargs["deep"] for c in completed.call_args_list))
            dispatch.assert_called_once()

    def test_bad_round_label_or_dependency_rejected(self):
        for label, r in (("unknown",0),("binary",6),("binary",True)):
            with self.assertRaises(ValueError):
                ctl.version(label,r)
        with patch.object(ctl.subprocess,"run") as run:
            with self.assertRaises(ValueError):
                ctl.submit("workers",0,"unknown")
            run.assert_not_called()

    def test_worker_and_coordinator_resource_contracts(self):
        exp = Path(__file__).resolve().parents[1]
        worker = (exp / "scripts/verge_control_workers.sbatch").read_text()
        coordinator = (exp / "scripts/verge_control_coordinate.sbatch").read_text()
        self.assertIn("--gpus=b200:1",worker)
        self.assertIn("--mem=32gb",worker)
        self.assertIn("--cpus-per-task=8",worker)
        self.assertNotIn("--gpus",coordinator)
        self.assertIn("--mem=4gb",coordinator)
        self.assertIn("verge_control_prepare.py",worker)
        self.assertIn("verge_control_train.py",worker)
        self.assertNotIn("verge_round_train.py",worker)


if __name__ == "__main__":
    unittest.main()
