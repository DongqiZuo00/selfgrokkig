# VERGE experiment archive

This repository preserves the experiment data, reports, code, configurations, and provenance for the VERGE work in the `self grok` workspace. Upload destination was explicitly supplied by the project owner on 2026-09-08.

## Current result

The three September 8 development rounds completed. Across the twelve Solver branches, 786,432 native training-generation tokens produced one actual binary-reward optimizer update, on an identity task. No target exact-pass or positive curriculum Gamma was observed. The pipeline execution is verified; method effectiveness and the complete long-running VERGE research scope remain unverified.

The latest diagnostic independently recomputes 768 unique physical selection-evaluation programs and 27,648 case executions. Every target case exact-pass is zero. Historical reports are preserved as historical records; their claims are not silently promoted to findings of the current design.

## Read the current reports

- [Result analysis](snapshots/local/outputs/selfgrok/verge_minimal_20260908/RESULT_ANALYSIS.md)
- [Method review](snapshots/local/outputs/selfgrok/verge_minimal_20260908/METHOD_REVIEW.md)
- [Implementation audit](snapshots/local/outputs/selfgrok/verge_minimal_20260908/IMPLEMENTATION_AUDIT.md)
- [Next research action](snapshots/local/outputs/selfgrok/verge_minimal_20260908/NEXT_ACTION.md)
- [Machine-readable experiment result](snapshots/local/outputs/selfgrok/verge_minimal_20260908/EXPERIMENT_RESULT.json)
- [Original design document](snapshots/local/outputs/selfgrok/verge_review_20260908/evidence/VERGE_own_method_design.md)

## Archive status

The initial upload contains the existing local snapshot, including raw records, logs, CPU tests, reports, scripts, configuration files, transfer archives, and the trained round-three g1 adapter. The remote project also contains historical experiment data and checkpoints. Their inventory and transfer are separate work in progress until `UPLOAD_STATUS.json` explicitly reports verification of the full selected scope.

Do not interpret this first commit as confirmation that every remote checkpoint is already uploaded. Check the manifests for the exact file inventory, sizes, SHA256 values, and any exclusions. Cached evaluation aliases are preserved but are not additional model samples.

## Preservation and reproduction

Files under `snapshots/` preserve their original contents. Original absolute paths inside logs and reports describe the execution environment and may not be runnable on another machine. The relative archive paths and manifests locate their preserved copies.

The current reports contain reproduction commands and their original execution logs. Running training is a separate action and requires a verified environment and compute budget; this repository does not automatically launch jobs or workflows.

The backbone is an external dependency; the saved experiment adapters and their source references are identified separately.
