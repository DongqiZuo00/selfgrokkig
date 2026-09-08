"""Pure recovery safety tests; no Slurm jobs, model samples or hashes."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_search_recover_followon as recovery


class RecoveryTests(unittest.TestCase):
    def args(self, **changes):
        root = Path(tempfile.gettempdir())/'verge_recovery_fixture'
        exp = root/'experiments/has_transfer_witness'
        expected = recovery.expected_arguments(exp)
        values = dict(original=dict(arguments=list(expected), confirmed_job_id=None),
            rejected=dict(arguments=list(expected), returncode=1, stdout='',
                          stderr='sbatch: error: Batch job submission failed: Job dependency problem\n'),
            expected=expected, accounting='41333225|COMPLETED|0:0\n41333225.batch|COMPLETED|0:0\n',
            live_returncode=1, live_stderr='slurm_load_jobs error: Invalid job id specified\n',
            queue='77|outside-project|'+str(root.parent/'another_project')+'\n', matching_history='', work_root=root)
        values.update(changes)
        return values

    def test_only_expired_dependency_removed(self):
        a=self.args()
        self.assertEqual(recovery.released_arguments(**a), [x for x in a['expected'] if x!='--dependency=afterok:41333225'])

    def test_failed_missing_duplicate_accounting_blocked(self):
        for s in ('', '41333225|FAILED|1:0', '41333225|COMPLETED|1:0', '41333225.batch|COMPLETED|0:0',
                  '41333225|RUNNING|0:0', '41333225|COMPLETED|0:0\n41333225|COMPLETED|0:0'):
            with self.subTest(s=s), self.assertRaises(RuntimeError):
                recovery.released_arguments(**self.args(accounting=s))

    def test_live_or_uncertain_lookup_blocked(self):
        for changes in (dict(live_returncode=0),dict(live_returncode=255),dict(live_stderr='Network timeout')):
            with self.assertRaises(RuntimeError):recovery.released_arguments(**self.args(**changes))

    def test_modified_or_confirmed_original_blocked(self):
        for changes in (dict(arguments=['sbatch']),dict(confirmed_job_id='555')):
            a=self.args();a['original'].update(changes)
            with self.assertRaises(RuntimeError):recovery.released_arguments(**a)

    def test_rejection_must_be_certified(self):
        for changes in (dict(returncode=0),dict(stdout='555'),dict(stderr='timeout'),dict(arguments=[])):
            a=self.args();a['rejected'].update(changes)
            with self.assertRaises(RuntimeError):recovery.released_arguments(**a)

    def test_duplicate_history_and_search_name_blocked(self):
        for changes in (dict(matching_history='555|COMPLETED|0:0'),dict(queue='555|vs-r0-prepare|/outside')):
            with self.assertRaises(RuntimeError):recovery.released_arguments(**self.args(**changes))

    def test_in_scope_or_unknown_ownership_blocked(self):
        a=self.args()
        for queue in ('555|other|'+str(a['work_root']/'experiments/other'), '555|other|(null)', '555|other'):
            with self.assertRaises(RuntimeError):recovery.released_arguments(**self.args(queue=queue))

    def test_existing_registry_or_journal_never_resubmitted(self):
        for file in ('jobs/round_00.json','submission_intents/round_00_prepare_expired_followon_recovery.json','active_wave.json'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p=root/file;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
                with patch.object(recovery.ctl,'ROOT',root),patch.object(recovery.ctl,'dispatch_lock',lambda:nullcontext()), \
                     patch.object(recovery.ctl,'predecessor_finished',return_value='41333225'), \
                     patch.dict(recovery.os.environ,VERGE_BOOK_SUITE='verge_book_v2',VERGE_SEARCH_SUITE='verge_search_v1'), \
                     patch.object(recovery,'command') as command:
                    with self.assertRaisesRegex(RuntimeError,'already attempted'):recovery.recover()
                    command.assert_not_called()


if __name__=='__main__':unittest.main()
