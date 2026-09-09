"""Verify preserved bytes in the commit whose ID was observed on GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--remote-head', required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--scope', default='local snapshot only; remote project archive tracked separately')
args = ap.parse_args()
head = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
assert head == args.remote_head, 'Local HEAD differs from observed GitHub HEAD'
manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
proc = subprocess.Popen(['git', '-C', str(args.repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
total = 0
try:
    for entry in manifest['files']:
        path = entry['destination_path']
        assert '\n' not in path and '\r' not in path
        proc.stdin.write(f'{head}:{path}\n'.encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().strip().split()
        assert len(header) == 3 and header[1] == 'blob', (path, header)
        size = int(header[2])
        assert size == entry['bytes'], path
        remaining = size
        digest = hashlib.sha256()
        while remaining:
            chunk = proc.stdout.read(min(remaining, 8 * 1024 * 1024))
            assert chunk, path
            digest.update(chunk)
            remaining -= len(chunk)
        assert proc.stdout.read(1) == b'\n', path
        source_sha = entry.get('source_sha256', entry.get('sha256'))
        assert digest.hexdigest() == source_sha == entry['destination_sha256'], path
        total += size
    proc.stdin.close()
    assert proc.wait() == 0
except BaseException:
    proc.kill()
    proc.wait()
    raise
result = {'status': 'PASS', 'repository': 'https://github.com/DongqiZuo00/selfgrokkig',
          'observed_remote_commit': head, 'snapshot_files_verified': len(manifest['files']),
          'snapshot_bytes_verified': total, 'verification': 'Every committed Git blob SHA256 matches source manifest; commit ID equals independently observed GitHub main ref.',
          'scope': args.scope}
with args.output.open('x', encoding='utf-8') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps(result))
