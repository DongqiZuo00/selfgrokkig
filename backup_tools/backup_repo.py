"""Offline, manifest-driven work backup. No network, GPU, or job-control calls.

plan freezes regular-file paths from a supplied inventory. pack captures those
paths only, groups ordinary files by directory, deduplicates prehashed checkpoint
content, and writes deterministic gzip streams split into <=1 GiB assets.
Changes and omissions remain explicit; this is not an atomic filesystem snapshot.
"""
from __future__ import annotations
import argparse
import collections
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import threading

SCHEMA = 'selfgrok-offline-backup-v1'
EXCLUDED = ('.git', 'outputs/github_archive_20260908')
WEIGHTS = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt'}
CHUNK = 1024 * 1024
MAX_PART = 1024 ** 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(CHUNK), b''):
            h.update(data)
    return h.hexdigest()


def cap_address_space(gib):
    if os.name == 'posix':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        value = gib * 1024 ** 3
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        if soft != resource.RLIM_INFINITY:
            value = min(value, soft)
        resource.setrlimit(resource.RLIMIT_AS, (value, hard))


def child_memory_limit():
    cap_address_space(4)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False).encode() + b'\n')


def relative_path(value):
    require(isinstance(value, str) and value and '\\' not in value and '\0' not in value,
            'invalid relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and ':' not in value and all(x not in ('', '.', '..') for x in value.split('/')),
            'absolute, noncanonical, or traversal path rejected: ' + value)
    return path.as_posix()


def excluded(value, exclusions=EXCLUDED):
    return any(value == p or value.startswith(p + '/') for p in exclusions)


def included(value, prefixes):
    return not prefixes or any(value == p or value.startswith(p + '/') for p in prefixes)


def inside(root, value, *, existing=False):
    value = relative_path(value)
    target = root.joinpath(*PurePosixPath(value).parts)
    require(target.resolve().is_relative_to(root.resolve()), 'path escapes root: ' + value)
    current = root
    for component in PurePosixPath(value).parts:
        current = current / component
        require(not current.is_symlink(), 'symlink refused: ' + value)
    if existing:
        require(target.is_file() and stat.S_ISREG(target.stat().st_mode), 'not a regular file: ' + value)
    return target


def state(st):
    # Windows fd/path ctime can differ for the same unchanged file; on POSIX it
    # is useful change metadata and remains part of all consistency comparisons.
    return {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns if os.name == 'posix' else None,
            'inode': st.st_ino, 'device': st.st_dev, 'mode': stat.S_IMODE(st.st_mode)}


def checkpoint(path):
    return path.startswith('checkpoints/') or PurePosixPath(path).suffix.lower() in WEIGHTS


def make_plan(inventory_path, output, include_prefixes=()):
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf-8'))
    require(isinstance(inventory.get('repository'), str) and isinstance(inventory.get('work'), str), 'inventory roots missing')
    require(Path(inventory['repository']).resolve().is_relative_to(Path(inventory['work']).resolve()), 'repository outside declared selfgrok work root')
    prefixes = sorted(set(relative_path(x) for x in include_prefixes))
    files, seen, ignored = [], set(), []
    for entry in inventory['files']:
        path = relative_path(entry['path'])
        require(path not in seen, 'duplicate inventory path: ' + path)
        seen.add(path)
        require(type(entry['bytes']) is int and entry['bytes'] >= 0 and type(entry['mtime_ns']) is int, 'invalid inventory metadata')
        if excluded(path) or not included(path, prefixes):
            ignored.append({'path': path, 'bytes': entry['bytes']})
            continue
        files.append({'path': path, 'bytes': entry['bytes'], 'mtime_ns': entry['mtime_ns'], 'checkpoint': checkpoint(path)})
    files.sort(key=lambda x: x['path'])
    env_cache = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
    groups = collections.defaultdict(lambda: {'files': 0, 'bytes': 0})
    for item in files:
        parts = PurePosixPath(item['path']).parts
        group = '/'.join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) > 1 else '[root files]'
        groups[group]['files'] += 1
        groups[group]['bytes'] += item['bytes']
        for i, component in enumerate(parts[:-1]):
            if component.lower() in {'env', 'envs', 'venv', '.venv', 'cache', '.cache', '__pycache__', 'node_modules'}:
                key = '/'.join(parts[:i + 1])
                env_cache[key]['files'] += 1
                env_cache[key]['bytes'] += item['bytes']
                break
    plan = {'schema': SCHEMA, 'kind': 'plan', 'created_utc': now(), 'source_root': inventory['repository'],
            'work_root': inventory['work'], 'inventory_sha256': digest_file(inventory_path),
            'scope': 'Explicit bounded prefix subset' if prefixes else 'Entire recorded project regular-file allowlist; no scientific-family filtering.',
            'included_prefixes': prefixes,
            'exclusions': list(EXCLUDED), 'git_history_backup': 'Separate git bundle, not implemented by this offline data packer.',
            'environment_or_cache_directories_included_not_silently_excluded': dict(env_cache),
            'groups': dict(groups), 'files': files, 'file_count': len(files), 'bytes': sum(x['bytes'] for x in files),
            'checkpoint_files': sum(x['checkpoint'] for x in files),
            'checkpoint_bytes': sum(x['bytes'] for x in files if x['checkpoint']),
            'excluded_file_count': len(ignored), 'excluded_bytes': sum(x['bytes'] for x in ignored),
            'inventory_symlinks_not_followed': inventory.get('symlinks', []),
            'inventory_special_files_not_captured': inventory.get('special_files', [])}
    plan['plan_sha256'] = hashlib.sha256(dumps(plan)).hexdigest()
    write_new(output, plan)
    return {k: v for k, v in plan.items() if k not in ('files', 'groups')}


def read_plan(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    actual = plan.pop('plan_sha256')
    require(hashlib.sha256(dumps(plan)).hexdigest() == actual, 'plan digest mismatch')
    plan['plan_sha256'] = actual
    require(plan['schema'] == SCHEMA and plan['kind'] == 'plan', 'unexpected plan schema')
    paths = [relative_path(x['path']) for x in plan['files']]
    require(len(paths) == len(set(paths)) and all(not excluded(p, plan['exclusions']) for p in paths), 'invalid frozen allowlist')
    return plan


def read_hashes(path, source_root):
    records, errors, header, footer, sizes = {}, [], None, None, {}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            kind = row.get('record_type')
            if kind == 'header':
                require(header is None and not records, 'misplaced hash header')
                header = row
            elif kind == 'footer':
                require(footer is None, 'duplicate hash footer')
                footer = row
            elif kind == 'error':
                require(footer is None, 'hash row after footer')
                errors.append(row)
            elif kind == 'file':
                require(header is not None and footer is None, 'hash file outside header/footer')
                name = relative_path(row['path'])
                require(name not in records and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'invalid or duplicate hash record')
                require(type(row['bytes']) is int and row['bytes'] >= 0 and type(row['mtime_ns']) is int, 'invalid hash metadata')
                require(row['sha256'] not in sizes or sizes[row['sha256']] == row['bytes'], 'same SHA with different byte lengths')
                sizes[row['sha256']] = row['bytes']
                records[name] = row
            else:
                raise ValueError('unknown hash record type')
    require(header is not None and footer is not None, 'incomplete checkpoint hash listing: header/footer required')
    require(Path(header['root']).resolve() == source_root.resolve(), 'checkpoint hash root differs from plan')
    require(len(records) + len(errors) == header['expected_files'], 'checkpoint listing count differs from header')
    return records, errors


class SplitWriter:
    def __init__(self, output, basename, part_bytes):
        require(128 <= part_bytes <= MAX_PART, 'part size must be 128 bytes through 1 GiB')
        self.output, self.basename, self.limit = output, basename, part_bytes
        self.parts, self.handle, self.h, self.size = [], None, None, 0

    def _close_part(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.parts.append({'path': self.filename, 'bytes': self.size, 'sha256': self.h.hexdigest()})
            self.handle = None

    def write(self, data):
        total = len(data)
        while data:
            if self.handle is None:
                self.filename = f'{self.basename}.part{len(self.parts) + 1:06d}'
                self.handle = (self.output / self.filename).open('xb')
                self.h, self.size = hashlib.sha256(), 0
            length = min(len(data), self.limit - self.size)
            block, data = data[:length], data[length:]
            self.handle.write(block)
            self.h.update(block)
            self.size += length
            if self.size == self.limit:
                self._close_part()
        return total

    def flush(self):
        if self.handle:
            self.handle.flush()

    def close(self):
        self._close_part()


class CompressedTar:
    def __init__(self, output, name, part_bytes, compressor, level, threads):
        self.split = SplitWriter(output, name + ('.tar.zst' if compressor == 'zstd' else '.tar.gz'), part_bytes)
        self.errors, self.process, self.thread = [], None, None
        if compressor in ('pigz', 'zstd'):
            executable = shutil.which(compressor)
            require(executable is not None, compressor + ' requested but unavailable')
            command = [executable, '-n', f'-{level}', '-p', str(threads), '-c'] if compressor == 'pigz' else [executable, f'-{level}', f'-T{threads}', '-q', '-c', '--check']
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            preexec_fn=child_memory_limit if os.name == 'posix' else None)
            def pump():
                try:
                    for data in iter(lambda: self.process.stdout.read(CHUNK), b''):
                        self.split.write(data)
                except Exception as error:
                    self.errors.append(error)
                    if self.process.poll() is None:
                        self.process.terminate()
            self.thread = threading.Thread(target=pump, daemon=True)
            self.thread.start()
            self.writer = self.process.stdin
        else:
            self.writer = gzip.GzipFile(filename='', mode='wb', compresslevel=level, fileobj=self.split, mtime=0)
        self.tar = tarfile.open(fileobj=self.writer, mode='w|', format=tarfile.PAX_FORMAT)

    def add(self, path, member, size):
        info = tarfile.TarInfo(member)
        info.size, info.mode, info.mtime, info.uid, info.gid = size, 0o600, 0, 0, 0
        with path.open('rb') as handle:
            self.tar.addfile(info, handle)

    def close(self):
        self.tar.close()
        self.writer.close()
        if self.process:
            self.thread.join(timeout=60)
            require(not self.thread.is_alive(), 'compression output pump did not terminate')
            require(self.process.wait() == 0 and not self.errors, 'compression output failed')
            self.process.stdout.close()
        self.split.close()
        return self.split.parts

    def abort(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.thread:
            self.thread.join(timeout=10)
        for object in (self.tar, self.writer):
            try:
                object.close()
            except (OSError, ValueError):
                pass
        try:
            self.split.close()
        except OSError:
            pass


def capture(source, temporary=None):
    before = state(source.stat())
    h, copied = hashlib.sha256(), 0
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(source, flags), 'rb') as src, (temporary.open('xb') if temporary else nullcontext()) as dst:
        opened = state(os.fstat(src.fileno()))
        require(stat.S_ISREG(os.fstat(src.fileno()).st_mode), 'capture source is not regular')
        remaining = opened['bytes']
        while remaining:
            data = src.read(min(CHUNK, remaining))
            if not data:
                break
            if dst:
                dst.write(data)
            h.update(data)
            copied += len(data)
            remaining -= len(data)
        after_fd = state(os.fstat(src.fileno()))
    try:
        after = state(source.stat())
    except OSError:
        after = None
    stable = before == opened == after_fd == after and copied == opened['bytes']
    return {'sha256': h.hexdigest(), 'bytes': copied, 'source_state': opened,
            'stable_during_capture': stable, 'path_state_after_capture': after}


def directory_scan(root, exclusions, prefixes=()):
    directories, links, regular = [], [], set()
    for folder, names, files in os.walk(root, followlinks=False):
        parent = Path(folder)
        relative = parent.relative_to(root).as_posix()
        names[:] = sorted(name for name in names if not excluded((PurePosixPath(relative) / name).as_posix().removeprefix('./'), exclusions))
        if prefixes:
            names[:] = [name for name in names if any(
                included((parent / name).relative_to(root).as_posix(), (prefix,)) or
                prefix.startswith((parent / name).relative_to(root).as_posix() + '/') for prefix in prefixes)]
        for name in names[:]:
            if (parent / name).is_symlink():
                links.append((parent / name).relative_to(root).as_posix())
                names.remove(name)
        if parent != root:
            directories.append({'path': relative, 'mtime_ns': parent.stat().st_mtime_ns, 'mode': stat.S_IMODE(parent.stat().st_mode)})
        for name in files:
            path = (parent / name).relative_to(root).as_posix()
            if excluded(path, exclusions) or not included(path, prefixes):
                continue
            if (parent / name).is_symlink():
                links.append(path)
            elif (parent / name).is_file():
                regular.add(path)
    return directories, links, regular


def _pack(plan_path, hashes_path, output, *, part_bytes=MAX_PART, compressor='gzip', level=6, threads=1, text_scan_path=None, context):
    plan = read_plan(plan_path)
    root, output = Path(plan['source_root']).resolve(), Path(output).resolve()
    work = Path(plan['work_root']).resolve()
    require(root.is_dir() and root.is_relative_to(work) and output.is_relative_to(work), 'source/output must stay within declared selfgrok work root')
    require(not output.exists(), 'backup output must be a new directory')
    if output.is_relative_to(root):
        require(excluded(output.relative_to(root).as_posix(), plan['exclusions']), 'backup output inside source must be in an explicitly excluded archive directory')
    require(1 <= threads <= 4 and 1 <= level <= 9, 'invalid compression resource setting')
    if compressor == 'auto':
        compressor = 'zstd' if shutil.which('zstd') else 'pigz' if shutil.which('pigz') else 'gzip'
    hashes, hash_errors = read_hashes(hashes_path, root)
    text_hashes, text_errors = read_hashes(text_scan_path, root) if text_scan_path else ({}, [])
    require(not text_errors and not any(x.get('candidate_count', 0) for x in text_hashes.values()), 'text scan errors/candidates require review before packing')
    selected = {x['path'] for x in plan['files']}
    missing_hash = [x['path'] for x in plan['files'] if x['checkpoint'] and x['path'] not in hashes]
    extra_hash = sorted(p for p in set(hashes) - selected if included(p, plan.get('included_prefixes', [])))
    require(not missing_hash, f'checkpoint hash listing does not cover {len(missing_hash)} planned paths; refresh inventory/hash plan first')
    output.mkdir(parents=True)
    context['output'] = output
    journal = (output / 'recovery_journal.jsonl').open('xb')
    context['journal'] = journal
    def journal_event(kind, payload):
        journal.write(dumps({'kind': kind, 'payload': payload}) + b'\n')
        journal.flush()
    journal_event('started', {'plan_sha256': plan['plan_sha256'], 'source_root': str(root), 'started_utc': now()})
    staging = output / '_staging'
    staging.mkdir()
    groups = collections.defaultdict(list)
    for item in plan['files']:
        parts = PurePosixPath(item['path']).parts
        group = 'checkpoint_objects' if item['checkpoint'] else '/'.join(parts[:2]) if len(parts) > 2 else parts[0] if len(parts) > 1 else 'root_files'
        groups[group].append(item)
    directories, source_links, initial_files = directory_scan(root, plan['exclusions'], plan.get('included_prefixes', []))
    manifest = {'schema': SCHEMA, 'kind': 'backup', 'started_utc': now(), 'source_root': str(root), 'work_root': str(work),
                'plan_sha256': plan['plan_sha256'], 'plan_file_sha256': digest_file(plan_path), 'hash_listing_sha256': digest_file(hashes_path),
                'text_scan_sha256': digest_file(text_scan_path) if text_scan_path else None,
                'text_scan_binding': 'provided scanned paths must match archived content SHA' if text_scan_path else 'no credential scan binding requested',
                'exclusions': plan['exclusions'], 'part_max_bytes': part_bytes, 'compression': compressor,
                'included_prefixes': plan.get('included_prefixes', []),
                'compression_threads': threads if compressor in ('pigz', 'zstd') else 1,
                'cli_posix_address_space_caps_gib': {'python': 12, 'compressor_child': 4}, 'files': [], 'directories': directories,
                'groups': [], 'changes': [], 'missing': [], 'hash_errors': hash_errors,
                'hash_paths_outside_frozen_plan_not_added': extra_hash, 'source_symlinks_not_followed': source_links,
                'unplanned_source_files_at_start': sorted(initial_files - selected),
                'consistency_limit': 'Every source, including duplicate checkpoint aliases, is content-hashed during packing. This is not an atomic snapshot. All recorded sources are checked again at the end.'}
    manifest['root_directory_metadata'] = {'mtime_ns': root.stat().st_mtime_ns, 'mode': stat.S_IMODE(root.stat().st_mode)}
    context['manifest'] = manifest
    journal_event('manifest_header', {k: v for k, v in manifest.items() if k not in ('files', 'groups')})
    blobs = {}
    for number, (group, items) in enumerate(sorted(groups.items())):
        slug = re.sub('[^A-Za-z0-9_.-]', '_', group)[:80]
        name = f'{number:04d}_{slug}'
        stream = None
        members = []
        for index, item in enumerate(items):
            path = item['path']
            temporary = staging / 'capture.bin'
            try:
                source = inside(root, path, existing=True)
                observed = state(source.stat())
                expected = hashes.get(path) if item['checkpoint'] else None
                eligible = bool(expected and expected.get('eligible_for_content_dedup') and expected.get('stable_during_hash')
                                and expected.get('unchanged_since_listing') and observed['bytes'] == expected['bytes']
                                and observed['mtime_ns'] == expected['mtime_ns'])
                key = (expected['sha256'], expected['bytes']) if eligible else None
                reused = key is not None and key in blobs
                if reused:
                    # mtime/size alone do not prove a duplicate still has the
                    # previously hashed content (e.g. copy tools preserve mtime).
                    captured = capture(source)
                    key = captured['sha256'], captured['bytes']
                    reused = key in blobs
                if reused:
                    member_group, member_name = blobs[key]
                    verification = 'current_alias_bytes_rehashed_matches_stored_blob'
                    if path in text_hashes:
                        require((captured['sha256'], captured['bytes']) == (text_hashes[path]['sha256'], text_hashes[path]['bytes']), 'text scan SHA no longer matches: ' + path)
                else:
                    captured = capture(source, temporary)
                    key = (captured['sha256'], captured['bytes'])
                    if path in text_hashes:
                        require(key == (text_hashes[path]['sha256'], text_hashes[path]['bytes']), 'text scan SHA no longer matches: ' + path)
                    if item['checkpoint'] and key in blobs:
                        member_group, member_name = blobs[key]
                    else:
                        member_name = 'objects/' + key[0] if item['checkpoint'] else 'files/' + path
                        member_group = name
                        if stream is None:
                            stream = CompressedTar(output, name, part_bytes, compressor, level, threads)
                            context['active_stream'] = stream
                        try:
                            stream.add(temporary, member_name, captured['bytes'])
                        except (OSError, ValueError) as error:
                            raise RuntimeError('archive stream write failed; current group is uncommitted') from error
                        members.append({'member': member_name, 'bytes': captured['bytes'], 'sha256': captured['sha256']})
                        if item['checkpoint']:
                            blobs[key] = (name, member_name)
                    temporary.unlink()
                    verification = 'captured_bytes_hashed'
                record = {'path': path, 'bytes': captured['bytes'], 'sha256': captured['sha256'],
                          'mtime_ns': captured['source_state']['mtime_ns'], 'mode': captured['source_state']['mode'],
                          'group': member_group, 'member': member_name, 'checkpoint': item['checkpoint'],
                          'verification': verification, 'source_state': captured['source_state'],
                          'stable_during_capture': captured['stable_during_capture'],
                          'unchanged_since_inventory': observed['bytes'] == item['bytes'] and observed['mtime_ns'] == item['mtime_ns']}
                if expected:
                    record['checkpoint_hash_matches_capture'] = (captured['sha256'], captured['bytes']) == (expected['sha256'], expected['bytes'])
                manifest['files'].append(record)
                journal_event('captured_file', record)
                if not record['stable_during_capture'] or not record['unchanged_since_inventory'] or record.get('checkpoint_hash_matches_capture') is False:
                    manifest['changes'].append({'path': path, 'stage': 'capture',
                                                'stable_during_capture': record['stable_during_capture'],
                                                'unchanged_since_inventory': record['unchanged_since_inventory'],
                                                'checkpoint_hash_matches_capture': record.get('checkpoint_hash_matches_capture')})
            except (OSError, ValueError) as error:
                if isinstance(error, OSError) and error.errno in (5, 28, 122):
                    raise RuntimeError('storage I/O/quota failure; preserve partial evidence') from error
                if temporary.exists():
                    temporary.unlink()
                manifest['missing'].append({'path': path, 'error_type': type(error).__name__, 'reason': str(error)})
                journal_event('pending_or_missing_file', manifest['missing'][-1])
        if stream:
            parts = stream.close()
            manifest['groups'].append({'id': name, 'logical_directory': group, 'format': 'tar+zstd' if compressor == 'zstd' else 'tar+gzip', 'parts': parts, 'members': members})
            context['active_stream'] = None
            journal_event('completed_group', manifest['groups'][-1])
    staging.rmdir()
    for record in manifest['files']:
        try:
            end = state(inside(root, record['path'], existing=True).stat())
            if end != record['source_state']:
                manifest['changes'].append({'path': record['path'], 'stage': 'final_restat', 'source_changed_after_capture': True})
        except (OSError, ValueError):
            manifest['changes'].append({'path': record['path'], 'stage': 'final_restat', 'source_missing_or_symlink': True})
    _, final_links, final_files = directory_scan(root, plan['exclusions'], plan.get('included_prefixes', []))
    for directory in directories:
        try:
            if inside(root, directory['path']).stat().st_mtime_ns != directory['mtime_ns']:
                manifest['changes'].append({'path': directory['path'], 'stage': 'final_directory_restat', 'directory_changed': True})
        except (OSError, ValueError):
            manifest['changes'].append({'path': directory['path'], 'stage': 'final_directory_restat', 'directory_missing_or_symlink': True})
    manifest['unplanned_source_files_at_finish'] = sorted(final_files - selected)
    manifest['source_symlinks_not_followed'] = sorted(set(source_links + final_links))
    if root.stat().st_mtime_ns != manifest['root_directory_metadata']['mtime_ns']:
        manifest['changes'].append({'path': '.', 'stage': 'final_directory_restat', 'root_directory_changed': True})
    manifest['finished_utc'] = now()
    manifest['captured_file_count'] = len(manifest['files'])
    manifest['original_path_bytes_to_restore'] = sum(x['bytes'] for x in manifest['files'])
    manifest['unique_tar_member_bytes'] = sum(m['bytes'] for g in manifest['groups'] for m in g['members'])
    manifest['compressed_asset_bytes'] = sum(p['bytes'] for g in manifest['groups'] for p in g['parts'])
    manifest['complete_and_unchanged_since_inventory'] = not any((manifest['changes'], manifest['missing'], hash_errors,
        extra_hash, manifest['source_symlinks_not_followed'], manifest['unplanned_source_files_at_start'], manifest['unplanned_source_files_at_finish']))
    write_new(output / 'backup_manifest.json', manifest)
    write_new(output / 'ACCEPTANCE.json', {k: manifest[k] for k in ('captured_file_count', 'original_path_bytes_to_restore', 'unique_tar_member_bytes', 'compressed_asset_bytes', 'complete_and_unchanged_since_inventory')})
    journal_event('finished', {'complete_and_unchanged_since_inventory': manifest['complete_and_unchanged_since_inventory']})
    journal.close()
    return manifest


def pack(plan_path, hashes_path, output, **kwargs):
    context = {}
    try:
        return _pack(plan_path, hashes_path, output, context=context, **kwargs)
    except BaseException as error:
        stream = context.get('active_stream')
        if stream:
            stream.abort()
        failure = {'status': 'interrupted_or_failed', 'error_type': type(error).__name__, 'reason': str(error), 'at_utc': now()}
        manifest = context.get('manifest')
        if manifest:
            completed = {g['id'] for g in manifest['groups']}
            partial = dict(manifest)
            partial['files'] = [f for f in manifest['files'] if f['group'] in completed]
            partial['uncommitted_files_not_claimed_restorable'] = [f['path'] for f in manifest['files'] if f['group'] not in completed]
            partial['failure'] = failure
            partial['complete_and_unchanged_since_inventory'] = False
            partial['captured_file_count'] = len(partial['files'])
            try:
                write_new(context['output'] / 'PARTIAL_MANIFEST.json', partial)
            except OSError:
                pass
        if context.get('output'):
            try:
                write_new(context['output'] / 'FAILED.json', failure)
            except OSError:
                pass
        journal = context.get('journal')
        if journal and not journal.closed:
            try:
                journal.write(dumps({'kind': 'failed', 'payload': failure}) + b'\n')
                journal.flush()
            except OSError:
                pass
            journal.close()
        raise


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.handle = iter(paths), None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.handle is None:
                try:
                    self.handle = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            length = self.handle.readinto(buffer)
            if length:
                return length
            self.handle.close()
            self.handle = None

    def close(self):
        if self.handle:
            self.handle.close()
        super().close()


@contextmanager
def decompress(parts, format):
    with io.BufferedReader(PartsReader(parts)) as raw:
        if format == 'tar+gzip':
            with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
                yield stream
            return
        executable = shutil.which('zstd')
        require(format == 'tar+zstd' and executable is not None, 'zstd executable required for this backup')
        process = subprocess.Popen([executable, '-d', '-q', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   preexec_fn=child_memory_limit if os.name == 'posix' else None)
        errors = []
        def pump():
            try:
                for block in iter(lambda: raw.read(CHUNK), b''):
                    process.stdin.write(block)
            except (OSError, ValueError) as error:
                errors.append(error)
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            yield process.stdout
            # Consume gzip/tar-style trailing padding so the decoder can finish.
            for _ in iter(lambda: process.stdout.read(CHUNK), b''):
                pass
            thread.join()
            require(process.wait() == 0 and not errors, 'zstd decompression failed')
        finally:
            if process.poll() is None:
                process.terminate()
            process.stdout.close()
            thread.join()
            process.wait()


def restore(manifest_path, destination=None, *, allow_incomplete=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['schema'] == SCHEMA and manifest['kind'] == 'backup', 'invalid backup manifest')
    require(allow_incomplete or manifest['complete_and_unchanged_since_inventory'], 'backup has recorded changes/omissions; inspect manifest and explicitly allow incomplete recovery')
    archive_root = manifest_path.parent
    records, expected = {}, {}
    for directory in manifest['directories']:
        require(not excluded(relative_path(directory['path']), manifest['exclusions']), 'excluded directory refused')
    for record in manifest['files']:
        path = relative_path(record['path'])
        require(path not in records and not excluded(path, manifest['exclusions']), 'duplicate/excluded restore path')
        relative_path(record['member'])
        require(re.fullmatch('[0-9a-f]{64}', record['sha256']) and type(record['bytes']) is int and record['bytes'] >= 0, 'invalid restored content metadata')
        records[path] = record
        key = record['group'], record['member']
        row = expected.setdefault(key, {'sha256': record['sha256'], 'bytes': record['bytes'], 'paths': []})
        require((row['sha256'], row['bytes']) == (record['sha256'], record['bytes']), 'aliases disagree about content')
        row['paths'].append(path)
    for path in records:
        require(all(parent.as_posix() not in records for parent in PurePosixPath(path).parents if parent.as_posix() != '.'), 'file conflicts with parent directory')
    asset_paths = set()
    for group in manifest['groups']:
        require(group['format'] in ('tar+gzip', 'tar+zstd'), 'unsupported compression format')
        for part in group['parts']:
            path = inside(archive_root, part['path'], existing=True)
            require(str(path) not in asset_paths, 'duplicate compressed asset')
            asset_paths.add(str(path))
            require(path.stat().st_size == part['bytes'] <= MAX_PART and digest_file(path) == part['sha256'], 'asset size/SHA mismatch')
    target = Path(destination).resolve() if destination is not None else None
    if target:
        require(not target.exists(), 'restore destination must be a new directory')
        require(not target.is_relative_to(archive_root) and not archive_root.is_relative_to(target), 'restore target overlaps backup source')
        source_root = Path(manifest['source_root']).resolve()
        require(not target.is_relative_to(source_root) and not source_root.is_relative_to(target), 'restore target overlaps original source tree')
        target.mkdir(parents=True)
    seen, restored = set(), 0
    for group in manifest['groups']:
        parts = [inside(archive_root, p['path'], existing=True) for p in group['parts']]
        with decompress(parts, group['format']) as compressed:
            with tarfile.open(fileobj=compressed, mode='r|') as archive:
                for member in archive:
                    name = relative_path(member.name)
                    key = group['id'], name
                    require(member.isfile() and not member.issym() and not member.islnk(), 'archive symlink/hardlink/special entry refused')
                    require(key in expected and key not in seen, 'undeclared or duplicate tar member')
                    item = expected[key]
                    require(member.size == item['bytes'], 'tar member length differs from manifest')
                    h, count, handle, first = hashlib.sha256(), 0, None, None
                    if target:
                        first = inside(target, item['paths'][0])
                        first.parent.mkdir(parents=True, exist_ok=True)
                        handle = first.open('xb')
                    try:
                        with archive.extractfile(member) as content:
                            for data in iter(lambda: content.read(CHUNK), b''):
                                h.update(data)
                                count += len(data)
                                if handle:
                                    handle.write(data)
                    finally:
                        if handle:
                            handle.close()
                    require(count == item['bytes'] and h.hexdigest() == item['sha256'], 'decompressed content SHA mismatch')
                    if target:
                        for index, path in enumerate(item['paths']):
                            dest = inside(target, path)
                            if index:
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                with first.open('rb') as src, dest.open('xb') as dst:
                                    shutil.copyfileobj(src, dst, CHUNK)
                            record = records[path]
                            os.chmod(dest, record['mode'])
                            os.utime(dest, ns=(record['mtime_ns'], record['mtime_ns']))
                            require(dest.stat().st_mtime_ns == record['mtime_ns'], 'destination filesystem cannot preserve exact file mtime_ns')
                            restored += 1
                    seen.add(key)
    require(seen == set(expected), 'missing tar members')
    if target:
        for directory in sorted(manifest['directories'], key=lambda x: len(PurePosixPath(x['path']).parts), reverse=True):
            path = inside(target, directory['path'])
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, directory['mode'])
            os.utime(path, ns=(directory['mtime_ns'], directory['mtime_ns']))
        root_metadata = manifest.get('root_directory_metadata')
        if root_metadata:
            os.chmod(target, root_metadata['mode'])
            os.utime(target, ns=(root_metadata['mtime_ns'], root_metadata['mtime_ns']))
    return {'status': 'captured_content_verified', 'mode': 'restore' if target else 'verify_without_restore',
            'restored_files': restored, 'verified_unique_members': len(seen), 'verified_original_paths': len(records),
            'all_assets_sha_verified': True, 'all_decompressed_content_sha_verified': True,
            'backup_complete_and_unchanged_since_inventory': manifest['complete_and_unchanged_since_inventory']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--inventory', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--include-prefix', action='append', default=[], help='Explicit bounded real-data probe only; omit for the whole inventory plan.')
    packer = commands.add_parser('pack')
    packer.add_argument('--plan', type=Path, required=True)
    packer.add_argument('--checkpoint-sha', type=Path, required=True)
    packer.add_argument('--text-scan-sha', type=Path)
    packer.add_argument('--output', type=Path, required=True)
    packer.add_argument('--part-bytes', type=int, default=MAX_PART)
    packer.add_argument('--compressor', choices=('auto', 'gzip', 'pigz', 'zstd'), default='auto')
    packer.add_argument('--level', type=int, default=3)
    packer.add_argument('--threads', type=int, default=1)
    for name in ('restore', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--allow-incomplete', action='store_true')
        if name == 'restore':
            command.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    cap_address_space(12)
    if os.name == 'posix' and args.command == 'pack':
        import signal
        def terminate(signum, frame):
            raise KeyboardInterrupt('received termination signal; writing partial recovery evidence')
        signal.signal(signal.SIGTERM, terminate)
    if args.command == 'plan':
        result = make_plan(args.inventory, args.output, args.include_prefix)
    elif args.command == 'pack':
        result = pack(args.plan, args.checkpoint_sha, args.output, part_bytes=args.part_bytes,
                      compressor=args.compressor, level=args.level, threads=args.threads, text_scan_path=args.text_scan_sha)
        result = {k: v for k, v in result.items() if k not in ('files', 'groups', 'directories')}
    else:
        result = restore(args.manifest, getattr(args, 'destination', None), allow_incomplete=args.allow_incomplete)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command == 'pack' and not result['complete_and_unchanged_since_inventory']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
