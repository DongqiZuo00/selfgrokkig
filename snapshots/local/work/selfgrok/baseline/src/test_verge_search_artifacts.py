"""Synthetic gate tests. A successful mock is not a real completion receipt."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_verge_search_report import bundle_fixture
import verge_search_artifacts as gate


class SearchArtifactTests(unittest.TestCase):
    def test_synthetic_bundle_never_finalizes(self):
        bundle=bundle_fixture()
        with patch.object(gate.report.inputs,'collect',return_value=bundle),patch.object(gate.ctl,'read',return_value=bundle):
            with self.assertRaisesRegex(RuntimeError,'Synthetic'):gate.validate()

    def test_review_rejects_missing_synthetic_and_duplicate(self):
        valid={'synthetic':False,'main_figure_reviewed':True,'csv_previews_reviewed':list(gate.COLUMNS),
            'all_labels_and_values_readable':True,'latex_values_reviewed':True}
        for value in ({},dict(valid,synthetic=True),dict(valid,latex_values_reviewed=False),
                      dict(valid,csv_previews_reviewed=list(gate.COLUMNS)+['condition_profiles'])):
            with self.assertRaises(RuntimeError):gate.validate_visual_review(value,Path('nonexistent'))

    def test_review_is_bound_to_file_size_and_time_not_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);names=['search_trajectories.png','result_table.tex']+[f'{n}.preview.png' for n in gate.COLUMNS]
            value={'synthetic':False,'main_figure_reviewed':True,'csv_previews_reviewed':list(gate.COLUMNS),
                'all_labels_and_values_readable':True,'latex_values_reviewed':True,'reviewed_file_metadata':{}}
            for name in names:
                p=root/name;p.write_bytes(b'synthetic fixture only');s=p.stat()
                value['reviewed_file_metadata'][name]={'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
            gate.validate_visual_review(value,root)
            (root/names[0]).write_bytes(b'changed fixture')
            with self.assertRaisesRegex(RuntimeError,'changed'):gate.validate_visual_review(value,root)

    def test_mock_success_still_leaves_book_incomplete(self):
        bundle=bundle_fixture();bundle['synthetic']=False
        records={'report_bundle.json':bundle,'render_values.json':{'rounds':gate.report.summary_rows(bundle),'synthetic':False,'suite_complete':False},
            'core_render_receipt.json':{'synthetic':False,'figure_rendered':True,'markdown_rendered':True,'latex_rendered':True,
                'search_block_complete':False,'suite_complete':False},'artifact_visual_review.json':{}}
        texts={'RESULTS_SEARCH.md':gate.report.markdown(bundle),'result_table.tex':gate.report.latex_table(bundle)}
        with patch.object(gate.report.inputs,'collect',return_value=bundle),patch.object(gate.ctl,'read',side_effect=lambda p:records[p.name]), \
                patch.object(Path,'read_text',lambda p,**kw:texts[p.name]),patch.object(gate,'validate_csv',return_value={}), \
                patch.object(gate,'validate_png') as pictures,patch.object(gate,'validate_visual_review'):
            result=gate.validate()
        self.assertTrue(result['search_block_complete']);self.assertFalse(result['suite_complete'])
        self.assertEqual(result['branches'],24);self.assertEqual(result['loss_tokens'],12582912)
        self.assertTrue(result['required_matrix_remaining']);self.assertEqual(pictures.call_count,7)


if __name__=='__main__':unittest.main()
