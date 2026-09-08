# Names-only v2 preparation

`prepare_named_round.py` is a new entry; it leaves the v1 preparation, decoder, executor, scorer, jobs and checkpoints intact. It defaults to `benchmarks/generated/naming_round_v2`, screen `runs/naming_v2_41415498`, history `runs/round1_result`, and history plan `runs/round_prepare_41413994/plan.json`. Each path has an explicit CLI override. It does not submit a job.

CPU input review:

```sh
python runtime/prepare_named_round.py --dry-run --output runtime/named_inputs_review_NEW
python -m unittest discover -s runtime -p test_prepare_named_round.py -v
```

A dry-run checks the frozen target, catalogue, metadata and four validator fields through the existing executor loader. Missing history or screen is recorded as a pending dependency; the schema and baseline contract are still reviewable. It does not load a model or write READY. Existing output directories are never reused. Ten CPU tests passed on 2026-09-08; invented rollout records are confined to temporary fixture directories and are explicitly rejected by the real screen/history loaders.

When the launcher decides to run after the real naming screen completes, the same entry without `--dry-run` first creates a new unhinted baseline: the original initial Solver, eight frozen selection instances, 32 actual samples each, chunks of eight, cap 2048 and the existing grammar/sampling contract. No historical raw samples are reused across prompt views. Only this new measurement enters the current incumbent rho. The independent original Challenger then generates its actual JSON proposal using the unchanged nine-stage menu and schema. The plan retains G=3, L=3, B=65536 per branch, alpha=eta=0.25, q=4 and n_min=8. This preparation launches neither the four training branches nor Challenger RFT.

The completed old round is loaded with its actual exchange/plan hashes, Gamma, per-stage Delta including target-only tail and endpoint correction, archive and proposed curricula. It is explicitly labelled a different prompt view and diagnostic history. Its checkpoints, rho and archive are not current incumbents. The real local history read confirmed round1 used 262144 training tokens, made zero curriculum updates, had Gamma=0 with [0,0] intervals for all three courses, J+=empty, kappa=2 and target rate=0. Those results describe a failed training attempt, not successful transfer. `named_round_real_history_read.json` preserves the sourced values and hashes.

The v2 program-library index contains only complete-suite binary successes in the actual naming screen. Each entry binds its raw file, slot, program hash and suite hash. Raw rows must match the new catalogue prompts/tests; old prompts, modified source weights, inconsistent rewards/counts or incomplete suites fail input validation. No hint stage is admitted without its separate 512-rollout hint-regret test. Mixed-group counts are resource diagnostics only: they are not a fifth validator and no positive Gamma is required for admission.

`READY.json` is written only after actual new-view baseline and Challenger JSON generation and plan validation. It binds target/catalogue paths and hashes, source adapter/Challenger hashes, entry source, screen/history sources and the baseline contract. A new reviewed runtime contract must accompany subsequent branch jobs. The existing v1 wrapper's required-file list names v1 input files; a v2 contract must also bind every new v2 input, or a separately reviewed v2 wrapper should select its required input paths. Use a preparation directory beginning `runs/round_prepare_` if retaining that wrapper.

The eight-instance pilot and bounded catalogue remain development restrictions. This entry preserves the full research problem, task/test semantics, three stage-source interfaces and formal scope; it does not establish the formal predictions.
