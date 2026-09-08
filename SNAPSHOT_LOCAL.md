# Selfgrok local archive snapshot

Status: **PASS**. Completed UTC: 2026-09-08T23:20:22.905226+00:00.

Copied **2,752 files / 266,937,717 bytes** from `outputs/selfgrok` and `work/selfgrok` into `repository/snapshots/local/`, preserving those source-relative paths and 155 directories.

Each file was checked against the reviewed inventory, copied without modifying its source, and independently hashed at source and destination. Source content was hashed again after the copy. All file sizes and SHA256 values match. The source file and directory sets stayed unchanged during the copy.

The archive preparation directory `outputs/selfgrok/github_archive_20260908` is excluded to prevent recursive self-copying. Any link whose resolved target leaves the two authorised selfgrok source trees would also be excluded and recorded; none were present in this snapshot.

All source reports, raw logs, compressed archives, model weights and Python cache files are retained. Empty directories exist in this local snapshot; Git alone does not track empty directories. This operation made no commits or uploads.

Credential pattern review: 0 candidate locations; 0 withheld files. The review covered readable text and tar text members. Binary model/optimizer contents and nested compressed members were not semantically interpreted for credentials.

Manifest: `local_snapshot_manifest.json` (SHA256 `1cdc5dd3cfa76b9f418f6adf0bdbe1f29497d3abddef94e145021cb815d90275`). It lists every source path, destination path, size, source SHA256, destination SHA256, and post-copy source SHA256.

Source inventory SHA256: `71f733e6a852aa5ee6bd5b7f35ecf284f54b3875d57d0d59f7fdce781677d6bf`.

Reproduction command (an interrupted copy may be resumed; preexisting destination files must already match the source hash):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/ddong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'outputs/selfgrok/github_archive_20260908/local_snapshot_copy.py'
```

The accompanying `local_snapshot_copy.log` records progress and the final PASS summary. The script refuses to overwrite a completed manifest, log or report.
