# Frozen compact target view for the development round

`target_rows.json` contains the actual 128 training rows and the first eight original selection rows. Original IDs, all 36 tests, six classes, expectations, hints (`null`), and every field except `messages` are unchanged. Only the generic syntax/task wording is shorter; no reference solution or hint is supplied.

Top-level backend interface:

```text
manifest_id
prompt_view_sha256
source_train_manifest_sha256
train: [128 rows]
selection: [8 rows]
```

`source_train_manifest_sha256` identifies the original `target_train.jsonl` bytes, checked against the original benchmark manifest. It does not identify the compact view file itself. The latter file hash is recorded in `target_view_metadata.json.target_rows_file_sha256`.

`target_view_metadata.json.selection` can be copied into the round request. It includes the original eight `instance_ids`, 32 frozen sample slots `rollout_00`..`rollout_31`, the common prompt view hash, and every selection test's explicit class mapping. The first eight sample slots are the stage/P8 prefix of P32.

- `manifest_id`: `verge_prepend_rb_v1_20260908:selection_first8:de03eceec552de11`
- `prompt_view_sha256`: `6b1f0506a1ec26cd0e9fabe27ac3826b660b1701ab1b61c90941caee88d298f6`
- `target_rows.json` file SHA256: `a15c1a7a6919ba4b79dac244f8bb09f98e2abb94fc3982c175d963e6a2dfe3b1`
- `target_view_metadata.json` file SHA256: `29edeff3bfc7463b1936eaff7a52c9678a7725090103d23d07e04bfc9fb069f3`
- `stage_validation_evidence.json` file SHA256: `828524dbdc536a837c41e0130558ff260e985cacafdcf697604594bad97230cf`

All three file hashes were identical on local Windows and remote HiPerGator. The prompt view hash is the canonical JSON hash of the explicit `prompt_view_hash_payload`, using the same convention as `round_cli.sha`; it is not a byte hash of the metadata file.

`stage_validation_evidence.json` provides `stage_validations[stage_id]` and `specifications[stage_id]` for all nine catalogue candidates. Each validator result binds the exact spec hash, protected manifest tuples, zero overlap with actual protected inputs, and the existing CPU oracle test IDs/log hashes. These are genuine CPU executable/nonconstant/leakage checks; non-hint stages explicitly carry `hint_regret='NA'`. This file does not claim model trainability, target transfer, or certify a hinted stage without its own real paired probe.

The source script `../../freeze_pilot.py` permits an identical rerun, but refuses to overwrite any of its three frozen artifacts with changed bytes. Choosing the first eight selection rows is a fixed development subset, not a replacement for the formal 64-instance evaluation or the P1–P7 research scope.
