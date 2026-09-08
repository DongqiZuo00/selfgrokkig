"""Synthetic rendering tests only; no additional model sampling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import verge_search_report as report
from test_verge_search_report_values import values,allocation_fixture


def bundle_fixture():
    data=values();snapshot,registered=allocation_fixture()
    for row in data['endpoint_rows']:
        if row['endpoint']=='selected_completion' and row['metric']=='parse':
            row['observed_count']=row['round']+1;row['observed_rate']=row['observed_count']/64
    return {'analysis':data,'resources':snapshot,'hardware':report.inputs.resource_totals(snapshot,registered),
            'synthetic':True,'suite_complete':False}


class SearchReportTests(unittest.TestCase):
    def test_summary_keeps_selection_and_fresh_check_denominators_separate(self):
        rows=report.summary_rows(bundle_fixture());self.assertEqual(len(rows),6)
        self.assertEqual(rows[-1]['fresh_completion_parse_count'],6)
        self.assertEqual(rows[-1]['fresh_completion_target_full_count'],0)
        self.assertEqual(rows[-1]['selection_endpoint_target_draws'],512)
        self.assertEqual(rows[-1]['fresh_completion_draws'],64)
        self.assertEqual(sum(s['gpu_allocation_hours'] for s in rows),36.)

    def test_stale_hardware_totals_cannot_be_rendered(self):
        bundle=bundle_fixture();bundle['hardware']['totals']['gpu_allocation_hours']=0.
        with self.assertRaises(ValueError):report.summary_rows(bundle)

    def test_markdown_does_not_promote_parse_or_pool_draws(self):
        content=report.markdown(bundle_fixture())
        self.assertIn('SYNTHETIC TEST DATA',content);self.assertIn('6/64',content)
        self.assertEqual(content.count('| 0/512 | 0/64 |'),6)
        self.assertIn('12,582,912',content);self.assertIn('36.0000 allocated GPU-hours',content)
        self.assertIn('never changes',content);self.assertIn('remaining experiment-book matrix',content)
        self.assertIn('does not establish an impossible target',content)

    def test_latex_keeps_exact_counts_and_valid_boundaries(self):
        content=report.latex_table(bundle_fixture())
        self.assertIn('SYNTHETIC TEST DATA',content)
        self.assertIn('6 & Start & 0/512 & 0/64 & 6/64',content)
        self.assertEqual(content.count(r'\begin{tabular}'),1);self.assertEqual(content.count(r'\end{tabular}'),1)

    def test_real_label_is_not_silently_attached_to_synthetic_inputs(self):
        bundle=bundle_fixture();del bundle['synthetic']
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):report.render(bundle,Path(tmp))

    def test_core_render_does_not_mark_visual_review_or_block_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);report.render(bundle_fixture(),path)
            self.assertEqual((path/'search_trajectories.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            receipt=json.loads((path/'core_render_receipt.json').read_text())
            self.assertFalse(receipt['visual_review_complete']);self.assertFalse(receipt['search_block_complete'])
            self.assertFalse(receipt['csv_exported']);self.assertFalse(receipt['suite_complete'])


if __name__=='__main__':unittest.main()
