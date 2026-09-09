"""Small synthetic CPU fixtures only; no real experiment files are written."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

import backup_repo as b

FIXTURES = Path(__file__).resolve().parent.parent / 'cpu_fixtures'


class BackupFixture(unittest.TestCase):
    def setUp(self):
        FIXTURES.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=FIXTURES)
        self.work = Path(self.temp.name).resolve()
        self.root = self.work / 'experiments/project'
        self.root.mkdir(parents=True)
        self.contents = {'checkpoints/a/state.pt': random.Random(9).randbytes(4096),
                         'checkpoints/b/state.pt': random.Random(9).randbytes(4096),
                         'logs/failed_job.err': b'FAILED: retained raw log\n',
                         'data/heldout/results.json': b'{"raw": [0, 1, 0]}\n',
                         'outputs/run/random.binlog': random.Random(17).randbytes(17000),
                         'src/code.py': b'print("fixture")\n',
                         '.git/HEAD': b'excluded only git metadata\n',
                         'outputs/github_archive_20260908/do_not_recurse.txt': b'excluded archive output\n',
                         'env/cache/keep.dat': b'cache is included and reported\n'}
        for i, (name, data) in enumerate(self.contents.items()):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(1700000000000000000 + i * 1000000,) * 2)
        (self.root / 'empty/nested').mkdir(parents=True)
        self.inventory = self.work / 'inventory.json'
        self.plan = self.work / 'plan.json'
        files = [{'path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size,
                  'mtime_ns': p.stat().st_mtime_ns} for p in self.root.rglob('*') if p.is_file()]
        b.write_new(self.inventory, {'repository': str(self.root), 'work': str(self.work), 'files': files})
        b.make_plan(self.inventory, self.plan)
        self.hashes = self.work / 'hashes.jsonl'
        self.hash_rows = []
        for name in self.contents:
            if name.startswith('checkpoints/'):
                p = self.root / name
                self.hash_rows.append({'record_type': 'file', 'path': name, 'bytes': p.stat().st_size,
                    'mtime_ns': p.stat().st_mtime_ns, 'sha256': b.digest_file(p), 'stable_during_hash': True,
                    'unchanged_since_listing': True, 'eligible_for_content_dedup': True})
        self.write_hashes(self.hashes, self.hash_rows)

    def write_hashes(self, path, rows, footer=True):
        records = [{'record_type': 'header', 'root': str(self.root), 'expected_files': len(rows)}] + rows
        if footer:
            records.append({'record_type': 'footer'})
        path.write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')

    def tearDown(self):
        assert self.work.is_relative_to(FIXTURES.resolve())
        # Only the checked synthetic fixture subtree is recursively removed.
        self.temp.cleanup()

    def pack(self, name='backup', **kwargs):
        output = self.work / name
        result = b.pack(self.plan, self.hashes, output, part_bytes=512, **kwargs)
        return output, result

    def assert_roundtrip(self, output, manifest, destination='restored', allow_incomplete=False):
        target = self.work / destination
        result = b.restore(output / 'backup_manifest.json', target, allow_incomplete=allow_incomplete)
        self.assertEqual(result['verified_original_paths'], len(manifest['files']))
        for record in manifest['files']:
            path = target / record['path']
            self.assertEqual(b.digest_file(path), record['sha256'])
            self.assertEqual(path.stat().st_mtime_ns, record['mtime_ns'])
        self.assertTrue((target / 'empty/nested').is_dir())
        return target

    def test_roundtrip_bytes_paths_nanosecond_mtimes_and_checkpoint_aliases(self):
        output, manifest = self.pack()
        self.assertTrue(manifest['complete_and_unchanged_since_inventory'])
        target = self.assert_roundtrip(output, manifest)
        self.assertEqual((target / 'checkpoints/a/state.pt').read_bytes(), (target / 'checkpoints/b/state.pt').read_bytes())
        self.assertEqual(manifest['original_path_bytes_to_restore'] - manifest['unique_tar_member_bytes'], 4096)
        self.assertTrue(all(p['bytes'] <= 512 for g in manifest['groups'] for p in g['parts']))
        self.assertTrue(any(len(g['parts']) > 2 for g in manifest['groups']))
        self.assertFalse((target / '.git').exists())
        self.assertFalse((target / 'outputs/github_archive_20260908').exists())
        self.assertIn('env', json.loads(self.plan.read_text())['environment_or_cache_directories_included_not_silently_excluded'])
        self.assertEqual((target / 'env/cache/keep.dat').read_bytes(), self.contents['env/cache/keep.dat'])

    def test_identical_sources_produce_identical_compressed_asset_bytes(self):
        _, one = self.pack('first')
        _, two = self.pack('second')
        self.assertEqual([[p['sha256'] for p in g['parts']] for g in one['groups']],
                         [[p['sha256'] for p in g['parts']] for g in two['groups']])

    def test_changed_checkpoint_is_not_restored_as_unchanged_duplicate(self):
        changed = self.root / 'checkpoints/a/state.pt'
        changed.write_bytes(b'changed actual checkpoint')
        output, manifest = self.pack()
        self.assertFalse(manifest['complete_and_unchanged_since_inventory'])
        with self.assertRaises(ValueError):
            b.restore(output / 'backup_manifest.json', self.work / 'refused')
        target = self.assert_roundtrip(output, manifest, allow_incomplete=True)
        self.assertEqual((target / 'checkpoints/a/state.pt').read_bytes(), b'changed actual checkpoint')
        self.assertEqual((target / 'checkpoints/b/state.pt').read_bytes(), self.contents['checkpoints/b/state.pt'])

    def test_duplicate_alias_same_size_and_restored_mtime_rehashes_current_bytes(self):
        path = self.root / 'checkpoints/b/state.pt'
        old = path.stat()
        replacement = b'X' * old.st_size
        path.write_bytes(replacement)
        os.utime(path, ns=(old.st_mtime_ns, old.st_mtime_ns))
        output, manifest = self.pack()
        self.assertFalse(manifest['complete_and_unchanged_since_inventory'])
        target = self.assert_roundtrip(output, manifest, allow_incomplete=True)
        self.assertEqual((target / 'checkpoints/b/state.pt').read_bytes(), replacement)
        self.assertNotEqual((target / 'checkpoints/a/state.pt').read_bytes(), replacement)

    def test_final_restat_detects_change_after_capture_without_claiming_atomicity(self):
        original = b.capture
        def capture(source, temp=None):
            result = original(source, temp)
            if source.name == 'failed_job.err':
                source.write_bytes(b'job continued writing after capture\n')
            return result
        with mock.patch.object(b, 'capture', side_effect=capture):
            output, manifest = self.pack()
        self.assertTrue(any(x['stage'] == 'final_restat' for x in manifest['changes']))
        target = self.assert_roundtrip(output, manifest, allow_incomplete=True)
        self.assertEqual((target / 'logs/failed_job.err').read_bytes(), self.contents['logs/failed_job.err'])

    def test_inventory_and_restore_traversal_rejected(self):
        for path in ('../escape', '/absolute', 'C:/escape', 'a\\b', 'a/../b'):
            with self.assertRaises(ValueError):
                b.relative_path(path)
        output, manifest = self.pack()
        manifest['files'][0]['path'] = '../escape'
        b.write_new(output / 'bad_manifest.json', manifest)
        with self.assertRaises(ValueError):
            b.restore(output / 'bad_manifest.json', self.work / 'bad_restore')
        self.assertFalse((self.work / 'bad_restore').exists())

    def test_checkpoint_listing_requires_footer_and_sha_size_consistency(self):
        incomplete = self.work / 'incomplete.jsonl'
        self.write_hashes(incomplete, self.hash_rows, footer=False)
        with self.assertRaises(ValueError):
            b.read_hashes(incomplete, self.root)
        rows = [dict(x) for x in self.hash_rows]
        rows[1]['bytes'] += 1
        broken = self.work / 'broken.jsonl'
        self.write_hashes(broken, rows)
        with self.assertRaises(ValueError):
            b.read_hashes(broken, self.root)

    def test_corrupt_asset_rejected_before_creating_restore_destination(self):
        output, manifest = self.pack()
        part = output / manifest['groups'][0]['parts'][0]['path']
        data = part.read_bytes()
        part.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaises(ValueError):
            b.restore(output / 'backup_manifest.json', self.work / 'bad_restore')
        self.assertFalse((self.work / 'bad_restore').exists())

    def test_tar_symlink_member_is_refused(self):
        output, manifest = self.pack()
        group = manifest['groups'][0]
        member = group['members'][0]['member']
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as archive:
            item = tarfile.TarInfo(member)
            item.type, item.linkname = tarfile.SYMTYPE, '../../outside'
            archive.addfile(item)
        data = gzip.compress(raw.getvalue(), mtime=0)
        evil = output / 'malicious.part'
        evil.write_bytes(data)
        group['parts'] = [{'path': evil.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]
        b.write_new(output / 'malicious_manifest.json', manifest)
        with self.assertRaises(ValueError):
            b.restore(output / 'malicious_manifest.json', self.work / 'malicious_restore')
        self.assertFalse((self.work / 'outside').exists())

    def test_new_unlisted_file_is_recorded_and_not_silently_packed(self):
        (self.root / 'logs/new.err').write_bytes(b'new failed log')
        output, manifest = self.pack()
        self.assertIn('logs/new.err', manifest['unplanned_source_files_at_start'])
        self.assertFalse(manifest['complete_and_unchanged_since_inventory'])
        self.assertNotIn('logs/new.err', {x['path'] for x in manifest['files']})

    def test_text_scan_hash_binds_archived_bytes(self):
        path = self.root / 'logs/failed_job.err'
        rows = [{'record_type': 'file', 'path': 'logs/failed_job.err', 'bytes': path.stat().st_size,
                 'mtime_ns': path.stat().st_mtime_ns, 'sha256': b.digest_file(path), 'candidate_count': 0}]
        scan = self.work / 'text_scan.jsonl'
        self.write_hashes(scan, rows)
        path.write_bytes(b'new bytes not covered by prior credential scan')
        _, manifest = self.pack(text_scan_path=scan)
        self.assertTrue(any(x['path'] == 'logs/failed_job.err' and 'text scan SHA' in x['reason'] for x in manifest['missing']))
        self.assertNotIn('logs/failed_job.err', {x['path'] for x in manifest['files']})

    def test_verify_reads_all_assets_and_members_without_restoring(self):
        output, manifest = self.pack()
        result = b.restore(output / 'backup_manifest.json')
        self.assertEqual(result['restored_files'], 0)
        self.assertEqual(result['verified_original_paths'], manifest['captured_file_count'])

    def test_failed_group_keeps_journal_and_partial_manifest_without_claiming_group(self):
        close = b.CompressedTar.close
        def fail(stream):
            close(stream)
            raise OSError('synthetic interruption after group output')
        with mock.patch.object(b.CompressedTar, 'close', new=fail):
            with self.assertRaises(OSError):
                self.pack()
        output = self.work / 'backup'
        self.assertTrue(list(output.glob('*.part*')))
        self.assertTrue((output / 'recovery_journal.jsonl').is_file())
        self.assertTrue((output / 'FAILED.json').is_file())
        partial = json.loads((output / 'PARTIAL_MANIFEST.json').read_text())
        self.assertFalse(partial['complete_and_unchanged_since_inventory'])
        self.assertEqual(partial['groups'], [])
        self.assertEqual(partial['files'], [])
        self.assertTrue(partial['uncommitted_files_not_claimed_restorable'])

    def test_restore_cannot_write_into_original_source_tree(self):
        output, manifest = self.pack()
        with self.assertRaises(ValueError):
            b.restore(output / 'backup_manifest.json', self.root / 'new_restore')
        self.assertFalse((self.root / 'new_restore').exists())

    def test_explicit_bounded_prefix_probe_does_not_capture_other_directories(self):
        subset = self.work / 'subset_plan.json'
        b.make_plan(self.inventory, subset, ['logs'])
        output = self.work / 'subset_backup'
        manifest = b.pack(subset, self.hashes, output, part_bytes=512)
        self.assertTrue(manifest['complete_and_unchanged_since_inventory'])
        self.assertEqual([r['path'] for r in manifest['files']], ['logs/failed_job.err'])
        self.assertEqual(manifest['unplanned_source_files_at_start'], [])
        b.restore(output / 'backup_manifest.json', self.work / 'subset_restored')

    @unittest.skipUnless(shutil.which('zstd'), 'zstd executable unavailable; gzip fixtures still run')
    def test_real_zstd_roundtrip(self):
        output, manifest = self.pack(compressor='zstd', level=3, threads=2)
        self.assertTrue(all(x['format'] == 'tar+zstd' for x in manifest['groups']))
        self.assert_roundtrip(output, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
