"""Synthetic artifact contracts. These tests never mark a real block complete."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_followon_artifacts as artifacts
import verge_followon_report as report
from test_verge_followon_report import data_fixture,hardware_fixture


class FollowonArtifactTests(unittest.TestCase):
    def test_latex_keeps_censoring_and_null_concordance(self):
        data=data_fixture();snapshot,registered=hardware_fixture()
        text=report.latex_tables(data,report.resource_totals(snapshot,registered),synthetic=True)
        self.assertIn('SYNTHETIC TEST DATA',text)
        self.assertEqual(text.count('& 18 & 18 & 0 & 18 &'),4)
        self.assertEqual(text.count('& 24 & 0.0000 & 0.0000 & 0.0000 & N/A'),3)
        self.assertIn('18,874,368',text);self.assertIn('not six training trials',text)

    def test_latex_rejects_incomplete_or_forged_analysis(self):
        data=data_fixture();data['endpoint_rows'].pop()
        with self.assertRaises(ValueError):report.latex_tables(data,{})

    def test_synthetic_stale_or_missing_bundle_cannot_finalize(self):
        data,snapshot,hardware={}, {'allocations':[]}, {}
        for bundle in ({'synthetic':True},{'synthetic':False,'analysis':{'stale':True}}):
            with patch.object(artifacts.report,'collect',return_value=(data,snapshot,hardware)), \
                    patch.object(artifacts.ctl,'read',return_value=bundle):
                with self.assertRaisesRegex(RuntimeError,'Synthetic, stale'):artifacts.validate()

    def test_missing_duplicated_or_synthetic_review_is_rejected(self):
        review={'synthetic':False,'main_figure_reviewed':True,
            'csv_previews_reviewed':list(artifacts.COLUMNS),'all_labels_and_values_readable':True,'latex_values_reviewed':True}
        variants=[{},dict(review,synthetic=True),dict(review,latex_values_reviewed=False),
            dict(review,csv_previews_reviewed=list(artifacts.COLUMNS)+[next(iter(artifacts.COLUMNS))])]
        for candidate in variants:
            with self.assertRaises(RuntimeError):artifacts.validate_visual_review(candidate,Path('no-files'))

    def test_review_requires_unchanged_current_files_without_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);names=['followon_summary.png','result_tables.tex']+[f'{n}.preview.png' for n in artifacts.COLUMNS]
            review={'synthetic':False,'main_figure_reviewed':True,'csv_previews_reviewed':list(artifacts.COLUMNS),
                'all_labels_and_values_readable':True,'latex_values_reviewed':True,'reviewed_file_metadata':{}}
            for name in names:
                path=root/name;path.write_bytes(b'synthetic-review-contract-only')
                s=path.stat();review['reviewed_file_metadata'][name]={'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
            artifacts.validate_visual_review(review,root)
            (root/names[0]).write_bytes(b'different synthetic revision')
            with self.assertRaisesRegex(RuntimeError,'changed'):artifacts.validate_visual_review(review,root)

    def test_block_completion_does_not_claim_full_book(self):
        text=Path(artifacts.__file__).read_text()
        self.assertIn('suite_complete=False',text);self.assertIn('required_matrix_remaining=True',text)
        self.assertIn('candidates=72,cohorts=24',text)

    def test_successful_gate_return_keeps_only_block_scope(self):
        data=data_fixture();snapshot,registered=hardware_fixture()
        hardware=report.resource_totals(snapshot,registered)
        bundle={'synthetic':False,'analysis':data,'resources':snapshot,'hardware':hardware}
        records={'report_bundle.json':bundle,'analysis.json':data,'report_values.json':hardware,
            'resource_accounting_final.json':snapshot,'artifact_visual_review.json':{'synthetic_test_stub':True}}
        texts={'RESULTS_FOLLOWON.md':report.report(data,hardware),'result_tables.tex':report.latex_tables(data,hardware)}
        with patch.object(artifacts.report,'collect',return_value=(data,snapshot,hardware)), \
                patch.object(artifacts.ctl,'read',side_effect=lambda p:records[p.name]), \
                patch.object(Path,'read_text',lambda p,**kwargs:texts[p.name]), \
                patch.object(artifacts,'validate_csv',return_value={'synthetic':False,'csv_tables_verified':7,
                    'followon_block_complete':False,'suite_complete':False}), \
                patch.object(artifacts,'validate_png') as pictures,patch.object(artifacts,'validate_visual_review') as review:
            result=artifacts.validate()
        self.assertTrue(result['followon_block_complete']);self.assertFalse(result['suite_complete'])
        self.assertTrue(result['required_matrix_remaining']);self.assertEqual(result['candidates'],72)
        self.assertEqual(result['loss_tokens'],18874368);self.assertEqual(pictures.call_count,8)
        review.assert_called_once()

    def test_changed_markdown_or_latex_is_rejected(self):
        data=data_fixture();snapshot,registered=hardware_fixture()
        hardware=report.resource_totals(snapshot,registered)
        bundle={'synthetic':False,'analysis':data,'resources':snapshot,'hardware':hardware}
        records={'report_bundle.json':bundle,'analysis.json':data,'report_values.json':hardware,'resource_accounting_final.json':snapshot}
        for name in ('RESULTS_FOLLOWON.md','result_tables.tex'):
            texts={'RESULTS_FOLLOWON.md':report.report(data,hardware),'result_tables.tex':report.latex_tables(data,hardware)}
            texts[name]+='synthetic corruption'
            with patch.object(artifacts.report,'collect',return_value=(data,snapshot,hardware)), \
                    patch.object(artifacts.ctl,'read',side_effect=lambda p:records[p.name]), \
                    patch.object(Path,'read_text',lambda p,**kwargs:texts[p.name]), \
                    patch.object(artifacts,'validate_csv',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'differs'):artifacts.validate()


if __name__=='__main__':unittest.main()
