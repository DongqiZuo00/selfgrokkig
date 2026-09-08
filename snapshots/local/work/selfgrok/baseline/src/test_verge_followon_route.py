"""Shared runtime route tests; no model imports or training requests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verge_round_core as core


class RouteTests(unittest.TestCase):
    def test_missing_manifest_is_not_implicit_permission(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'VERSION','verge_followon_v1_verge_r00_c1'), \
                patch.object(core,'CFG_PATH',Path(tmp)/'missing.json'):
            with self.assertRaisesRegex(RuntimeError,'frozen configuration'):core.config()

    def test_explicit_followon_environment_and_frozen_predecessors_required(self):
        name='verge_followon_v1_verge_r00_c1'
        with tempfile.TemporaryDirectory() as tmp:
            exp=Path(tmp);path=exp/'candidate.json';path.write_text(json.dumps({'protocol_version':name,'followon_only':True}))
            with patch.object(core,'VERSION',name),patch.object(core,'CFG_PATH',path),patch.object(core,'EXP',exp):
                with patch.dict(os.environ,{'VERGE_FOLLOWON_SUITE':''}):
                    with self.assertRaisesRegex(RuntimeError,'explicitly'):core.config()
                with patch.dict(os.environ,{'VERGE_FOLLOWON_SUITE':'verge_followon_v1'}):
                    with self.assertRaisesRegex(RuntimeError,'frozen plan'):core.config()

    def test_primary_route_never_calls_followon_gate(self):
        name='verge_book_v2_verge_r00'
        cfg={'protocol_version':name,'book_suite':'verge_book_v2','independent_prompt_streams':True}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'primary.json';path.write_text(json.dumps(cfg))
            with patch.object(core,'VERSION',name),patch.object(core,'CFG_PATH',path), \
                    patch.dict(os.environ,{'VERGE_BOOK_SUITE':'verge_book_v2'}), \
                    patch('verge_followon_protocol.validate_launch',side_effect=AssertionError('Primary should bypass follow-on')):
                self.assertEqual(core.config(),cfg)


if __name__=='__main__':unittest.main()
