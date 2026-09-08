# Conditional ignition preparation

New entry: `prepare_ignition_round.py`. Existing frozen runtime, methods, source checkpoints and running jobs remain intact. Default dependencies are `runs/round_prepare_named_41417194`, `runs/round2_result`, `benchmarks/generated/naming_round_v2` and `runs/naming_v2_41415498`.

```sh
python runtime/prepare_ignition_round.py --dry-run --output runtime/ignition_dry_run_NEW
python -m unittest discover -s runtime -p test_prepare_ignition_round.py -v
```

The actual preparation command omits `--dry-run` and uses a new output under selfgrok, normally `runs/round_prepare_ignition_JOBID` for the existing branch wrapper. This entry submits no jobs. It requires complete actual round2 history and confirms its curriculum update counter against the actual phase ledgers. If round2 already produced a curriculum update, this conditional intervention stops before loading Q.

The declared human-set proposal prior restricts only `g1[0]`: the real Challenger chooses `identity` or `append_R`, each observed as a 1/8 mixed group under the training's concise prompt. Other eight positions retain all nine current stage specifications. The schema expresses the first position with `prefixItems`; the actual output must satisfy it and the unchanged complete stage objects. Four stage validators remain exactly unchanged. The policy appears in context, request and READY; this is not evidence of unconstrained proposal superiority.

The initial Solver and complete base sampling contract are retained. The loader verifies the source plan/READY/fragment, physical checkpoint, 256 original records and 32 original raw generation files, including hashes and native samples. A copied fragment keeps all old raw paths, evidence IDs, hashes and records unchanged; its added `baseline_reuse` and new `execution` fields explicitly count zero newly generated baseline rollouts/tokens. `original_execution` retains the original measurement cost. No baseline model is loaded. Only the independent initial Challenger model is loaded for new JSON generation.

The frozen scorer totals referenced evidence, including the original 256 baseline records. Final reporting must distinguish this evidence total from new generation in the ignition round: subtract `baseline_reuse.reused_rollout_slots` and `original_evaluation_generated_tokens` for the incremental accounting. Do not relabel copied records as new observations or replace their physical source paths.

G3, L3, B65536 per branch, alpha=eta=.25, n_min8, binary updates and skips, common base, fresh optimizers, target-only tail, unhinted P8/P32 and frozen target tests remain unchanged. New Q generation is capped at two attempts of 8192 tokens. No formal four-branch training or RFT is launched by preparation, and there is no promise of a new mixed group or positive Gamma.

Ten local CPU tests cover the position-specific schema, retained eight-slot menu, real probe eligibility, strict raw/base reuse, and incomplete-history behavior. `verify_ignition_schema_cpu.py` additionally uses the existing real vLLM/xgrammar CPU worker to compile the schema, replay every token of two explicitly marked grammar fixtures including EOS, reject an invalid first stage, and validate actual source baseline reuse. It loads a tokenizer, not model weights, with CUDA visibility empty. Its fixtures never become Q output or experiment results.

The real CPU verification passed with xgrammar 0.2.4 and Transformers 4.57.6: 38 structured constants expanded, both legal fixtures (2163 and 2183 tokenizer tokens including EOS) passed every actual mask, and the disallowed first stage was rejected. The actual 256 baseline records and 32 original raw files passed source verification. Exact evidence is retained in `evidence/ignition_schema_cpu_20260908/SUMMARY.json` and `actual_baseline_reuse_check.json`. A separate real-path dry-run completed with only the not-yet-present `round2_result/summary.json` pending and no READY or model loading.
