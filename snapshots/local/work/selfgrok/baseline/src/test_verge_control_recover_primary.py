import unittest
from verge_control_recover_primary import released_arguments


class ExpiredPrimaryTests(unittest.TestCase):
    def args(self, **changes):
        original = ['sbatch', '--dependency=afterok:41317782', '--array=0-2%2']
        values = dict(original=original, expected=list(original), dependency='41317782',
                      accounting='41317782|COMPLETED|0:0\n41317782.batch|COMPLETED|0:0\n',
                      live_returncode=1, live_stderr='slurm_load_jobs error: Invalid job id specified',
                      queue='40669351|duj-github-snapshot\n', matching_history='')
        values.update(changes)
        return values

    def test_only_the_completed_dependency_is_removed(self):
        self.assertEqual(released_arguments(**self.args()), ['sbatch', '--array=0-2%2'])

    def test_failed_pending_unknown_or_duplicate_accounting_rejected(self):
        for text in ('', '41317782|FAILED|1:0', '41317782|RUNNING|0:0',
                     '41317782|COMPLETED|1:0', '41317782.batch|COMPLETED|0:0',
                     '41317782|COMPLETED|0:0\n41317782|COMPLETED|0:0'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                released_arguments(**self.args(accounting=text))

    def test_live_or_uncertain_slurm_lookup_rejected(self):
        for changes in ({'live_returncode': 0}, {'live_stderr': 'Network timeout'}):
            with self.assertRaises(RuntimeError):
                released_arguments(**self.args(**changes))

    def test_possible_duplicate_or_resource_overlap_rejected(self):
        for changes in ({'matching_history': '41399999|PENDING|0:0'},
                        {'queue': '41399999|vc-r0-workers'}):
            with self.assertRaises(RuntimeError):
                released_arguments(**self.args(**changes))

    def test_modified_intent_rejected(self):
        with self.assertRaises(RuntimeError):
            released_arguments(**self.args(original=['sbatch', '--array=0-99']))


if __name__ == '__main__':
    unittest.main()
