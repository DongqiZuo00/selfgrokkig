"""Synthetic bounded submission and recovery checks; no real Slurm commands."""
from contextlib import ExitStack,nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import verge_search_controller as ctl
import test_verge_search_worker_protocol as worker_tests
from test_verge_search_worker_protocol import write


class ControllerTests(unittest.TestCase):
    def fixture(self,exp):
        worker_tests.WorkerProtocolTests().fixture(exp)
        write(exp/'raw_results/verge_followon_v1/jobs/block.json',{'coordinator':'900'})
        write(exp/'raw_results/verge_search_v1/IMPLEMENTATION_READY.json',
            {'runtime_enabled':True,'worker_tests_passed':True,'round_tests_passed':True,'controller_tests_passed':True,
             'synthetic_fixture':True})

    def patches(self,exp):
        stack=ExitStack();stack.enter_context(patch.object(ctl,'EXP',exp))
        stack.enter_context(patch.object(ctl,'ROOT',exp/'raw_results/verge_search_v1'))
        stack.enter_context(patch.object(ctl,'dispatch_lock',lambda:nullcontext()))
        stack.enter_context(patch.dict(ctl.os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_SEARCH_SUITE':'verge_search_v1'}))
        return stack

    def test_predecessor_and_runtime_gate_before_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp)
            with self.patches(exp),patch.object(ctl.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'predecessor'):ctl.dispatch()
                self.fixture(exp);(ctl.ROOT/'IMPLEMENTATION_READY.json').unlink()
                with self.assertRaisesRegex(RuntimeError,'not ready'):ctl.dispatch()
                run.assert_not_called()

    def test_three_stage_dependencies_and_idempotent_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);self.fixture(exp)
            with self.patches(exp),patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(stdout=str(x)) for x in (1000,1001,1002)]) as run:
                result=ctl.dispatch();self.assertEqual(result['finish'],'1002')
                self.assertIn('--dependency=afterok:900',run.call_args_list[0].args[0])
                self.assertIn('--dependency=afterok:1000',run.call_args_list[1].args[0])
                self.assertIn('--array=0-3%2',run.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:1001',run.call_args_list[2].args[0])
                self.assertEqual(ctl.dispatch(),result);self.assertEqual(run.call_count,3)
                configs=[ctl.read(exp/'manifests'/(ctl.version(0,b)+'.json')) for b in range(4)]
                self.assertEqual(len({c['training_random_stream_id'] for c in configs}),4)
                self.assertEqual(len({c['random_stream_id'] for c in configs}),1)

    def test_lost_submission_reply_cannot_duplicate_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);self.fixture(exp)
            with self.patches(exp),patch.object(ctl.subprocess,'run',return_value=SimpleNamespace(stdout='')) as run:
                with self.assertRaisesRegex(RuntimeError,'Uncertain'):ctl.dispatch()
                with self.assertRaisesRegex(RuntimeError,'Unresolved'):ctl.dispatch()
                self.assertEqual(run.call_count,1)

    def test_confirmed_intents_restore_outer_job_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);self.fixture(exp)
            with self.patches(exp),patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(stdout=str(x)) for x in (1000,1001,1002)]) as run:
                result=ctl.dispatch();(ctl.ROOT/'jobs/round_00.json').unlink()
                self.assertEqual(ctl.dispatch(),result);self.assertEqual(run.call_count,3)

    def test_following_round_inherits_selected_checkpoint_and_finish_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);self.fixture(exp)
            prior={'protocol_version':ctl.version(0),'search_round_complete':True,'suite_complete':False,
                   'selected':{'checkpoint':'checkpoints/verge_book_v2_initial_solver/resume_u0000'}}
            write(exp/'raw_results/verge_search_v1/rounds'/ctl.version(0)/'complete.json',prior)
            write(exp/'raw_results/verge_search_v1/jobs/round_00.json',{'finish':'1002'})
            with self.patches(exp),patch.object(ctl,'completed_round',side_effect=lambda r:prior if r==0 else None),patch.object(ctl.subprocess,'run',side_effect=[SimpleNamespace(stdout=str(x)) for x in (2000,2001,2002)]) as run:
                result=ctl.dispatch();self.assertEqual(result['round'],1)
                self.assertIn('--dependency=afterok:1002',run.call_args_list[0].args[0])
                self.assertEqual(ctl.read(exp/'manifests'/f'{ctl.version(1)}.json')['solver_start'],prior['selected']['checkpoint'])

    def test_six_round_limit_is_not_block_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);self.fixture(exp)
            for r in range(6):write(exp/f'raw_results/verge_search_v1/jobs/round_{r:02d}.json',{'finish':str(100+r)})
            with self.patches(exp),patch.object(ctl,'completed_round',return_value={'search_round_complete':True}),patch.object(ctl.subprocess,'run') as run:
                result=ctl.dispatch();self.assertTrue(result['search_rounds_finished'])
                self.assertFalse(result['search_block_complete']);self.assertFalse(result['suite_complete']);run.assert_not_called()

    def test_incomplete_round_never_advances(self):
        with patch.object(ctl,'completed_round',return_value=None),patch.object(ctl,'dispatch') as dispatch:
            with self.assertRaises(RuntimeError):ctl.coordinate(0)
            dispatch.assert_not_called()
        with patch.object(ctl,'completed_round',return_value={}) as completed,patch.object(ctl,'dispatch',return_value={}) as dispatch:
            ctl.coordinate(0);completed.assert_called_once_with(0,deep=True);dispatch.assert_called_once()

    def test_bounded_rounds_stages_and_resources(self):
        with patch.object(ctl.subprocess,'run') as run:
            for args in [('workers',6,'900'),('unknown',0,'900'),('prepare',0,'bad')]:
                with self.assertRaises(ValueError):ctl.submit(*args)
            run.assert_not_called()
        exp=Path(__file__).resolve().parents[1]
        for file in ('verge_search_workers.sbatch','verge_search_round.sbatch'):
            text=(exp/'scripts'/file).read_text()
            for option in ('--gpus=b200:1','--mem=32gb','--cpus-per-task=8','--time=02:00:00','set -euo pipefail'):
                self.assertIn(option,text)
            self.assertNotIn('verge_round_train.py',text)


if __name__=='__main__':unittest.main()
