"""Search route isolation: no model loading, sampling or Slurm calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_round_core as core
import test_verge_search_worker_protocol as worker_tests


class RouteTests(unittest.TestCase):
    def load(self,name):
        spec=importlib.util.spec_from_file_location('isolated_route_test',Path(__file__).resolve().parent/'verge_round_core.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(os.environ,{'VERGE_VERSION':name}):spec.loader.exec_module(module)
        return module

    def test_only_bounded_namespaces_and_expected_directories(self):
        for name in ['verge_search_v1_r06','verge_search_v1_r00_b4','verge_search_v1_r0','verge_search_v1_r00_c1']:
            with self.assertRaises(ValueError):self.load(name)
        r=self.load('verge_search_v1_r00');b=self.load('verge_search_v1_r00_b2')
        self.assertEqual(r.ROOT,r.EXP/'raw_results/verge_search_v1/rounds/verge_search_v1_r00')
        self.assertEqual(b.ROOT,b.EXP/'raw_results/verge_search_v1_r00_b2')
        for name in ['verge_book_v2_verge_r00','verge_control_v1_binary_r00','verge_followon_v1_verge_r00_c1']:
            old=self.load(name);self.assertEqual(old.ROOT,old.EXP/'raw_results'/name)

    def test_missing_configuration_is_not_implicit_permission(self):
        for name in ['verge_search_v1_r00','verge_search_v1_r00_b0']:
            with tempfile.TemporaryDirectory() as tmp,patch.object(core,'VERSION',name),patch.object(core,'CFG_PATH',Path(tmp)/'missing.json'):
                with self.assertRaisesRegex(RuntimeError,'frozen configuration'):core.config()

    def test_both_suite_variables_and_predecessors_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);worker,shared=worker_tests.WorkerProtocolTests().fixture(exp)
            round_cfg=shared['config']
            for cfg in (round_cfg,worker):
                name=cfg['protocol_version'];path=exp/'manifests'/(name+'.json');path.write_text(json.dumps(cfg))
                with patch.object(core,'VERSION',name),patch.object(core,'CFG_PATH',path),patch.object(core,'EXP',exp):
                    with patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_SEARCH_SUITE':''}):
                        with self.assertRaisesRegex(RuntimeError,'explicitly'):core.config()
                    with patch.dict(os.environ,{'VERGE_BOOK_SUITE':'','VERGE_SEARCH_SUITE':'verge_search_v1'}):
                        with self.assertRaisesRegex(RuntimeError,'explicitly'):core.config()
                    with patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2','VERGE_SEARCH_SUITE':'verge_search_v1'}):
                        self.assertEqual(core.config(),cfg)

    def test_primary_route_never_calls_search_gates(self):
        name='verge_book_v2_verge_r00';cfg={'protocol_version':name,'book_suite':'verge_book_v2','independent_prompt_streams':True}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'primary.json';path.write_text(json.dumps(cfg))
            with patch.object(core,'VERSION',name),patch.object(core,'CFG_PATH',path),patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2'}), \
                    patch('verge_search_worker_protocol.validate_launch',side_effect=AssertionError('wrong worker gate')), \
                    patch('verge_search_worker_protocol.validate_round_launch',side_effect=AssertionError('wrong round gate')):
                self.assertEqual(core.config(),cfg)


if __name__=='__main__':unittest.main()
