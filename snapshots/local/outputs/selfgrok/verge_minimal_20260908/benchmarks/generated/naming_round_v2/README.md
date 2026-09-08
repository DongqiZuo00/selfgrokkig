# Naming round view v2 — frozen CPU inputs

This directory contains an independent candidate view for a later round. It does not submit or authorize a run. The current v1 round and its runtime contract remain unchanged.

- `target_rows.json`: the same 128 original train rows and first 8 selection rows. All IDs, ground truth, 36 cases per instance, six classes, and every row field other than `messages` are unchanged from `pilot_view_v1`.
- `target_view_metadata.json`: the new target prompt hash and versioned manifest, selection contracts, and source hashes. `selection` is ready for a later request; the original target train manifest SHA remains `89f79a4df80a5871db61a02cc4f36319e46300626fdcf6431e697923c13be585`.
- `catalogue.json`: the same nine stages, kinds, specifications, rows and variants. Every user prompt receives the identical paragraph already frozen in `naming_prompt_v2/screen_tasks.json`.
- `stage_validation_evidence.json`: reuses the original genuine CPU spec/test validation and its evidence IDs. New prompt hashes are explicitly bound in each validation's `details.prompt_view_binding`; no new model effectiveness is asserted.

The only prompt addition explains canonical names start, n0, n1, …, end and declaration syntax. It permits start/end/NONE/intermediate routes, including forward references and loops. It neither prescribes the first route nor supplies a task program, oracle, or topology.

Target prompt-view SHA256: `22706996b598b5961ea73149600d4405dbdfe6500c4ba9917dd5cfd31e813b4e`.

Stage prompt-view SHA256: `e0fe6234cc312e4b5f15e145d32b0ec34dde8d58b7793e7f7bba18e55f1f58cb`.

Manifest: `verge_prepend_rb_v1_20260908:selection_first8:naming_v2:19840cace70bd077`.

`benchmarks/freeze_naming_round_v2.py` freezes reproducibly and refuses to replace changed artifacts. Four CPU tests in `benchmarks/test_freeze_naming_round_v2.py` verify exact field/test preservation, all stage variants and genuine validation provenance, new identities with the real `execute_round.load_inputs` path, and immutable writes plus untouched v1 source hashes. The loader test is only a read-contract fixture, never a Challenger proposal or a submitted run plan.
