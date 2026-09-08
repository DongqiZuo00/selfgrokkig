# Backend review — 2026-09-08

Reviewed `runtime/hf_backend.py`, `runtime/prepare_round.py`, their calls into `runtime/execute_round.py`, `round/round_cli.py`, and the grammar bridge. This was a read-only implementation review; no GPU job was launched and no implementation was changed. The exact source hashes and new CPU checks are in `BACKEND_REVIEW_CPU.json`.

**Conclusion:** no execution-blocking bug or grammar-generation/loss policy mismatch was found in the reviewed snapshot. This supports attempting the bounded development round. It does not establish transfer, useful curricula, or the effectiveness of the full VERGE method.

## Verified connections

- **Actual model and independent adapters.** Both Solver and Challenger use the explicitly named local Ministral-3-3B-Instruct-2512-BF16 model. Adapter loading verifies the actual base-model path, rank 16, and model type. Solver and Q begin at the distinct audited `resume_u0000` adapters. Only LoRA parameters can train. Existing checkpoints are read; new checkpoint saves refuse an existing destination.
- **Native generated actions.** The DSL format prefix is prompt context. The verifier receives `prefix + decoded native suffix`; the loss and generated-token count use only actual suffix token IDs. The first generated EOS counts once; subsequent batch padding is removed. A cap-truncated suffix remains a real sampled action sequence; strict verification still determines its binary reward.
- **Same constrained policy in generation and update.** Each `generate()` call creates a fresh `HFGrammarLogitsProcessor` while reusing the initialized grammar worker. The sampler disables extra probability warpers. Before training, the backend checks the recorded grammar metadata against that same worker. Replay reconstructs each actual suffix's allowed-token sets; `masked_suffix_loss` normalizes logits over these sets, including EOS, with no loss on the prompt. It does not compute an unmasked loss on guided samples.
- **Group semantics.** One eight-rollout group shares one prompt and one verifier. The only learning reward is integer binary full pass. Partial class fractions are used for evaluation conditions, never advantages. Constant-reward groups do not step optimizer or scheduler. The mixed-group loss uses group-relative binary advantages and normalization by actual generated tokens. Dropout, KL beta, and weight decay are zero. Each new branch constructs a fresh optimizer at the common base.
- **Budget boundaries.** `run_phase` caps each generation so the phase cannot exceed its remaining generated-token quota. The remaining fewer-than-eight tokens are explicitly recorded as non-learning drain. EOS tokens count; the fixed prefix, prompts, and post-EOS batch pad do not. Intermediate/final probes have their own costs. Branch checkpoint aliases and fingerprints identify the actual weights after each phase.
- **Frozen baseline and probes.** Preproposal baseline sampling uses the frozen unhinted target view, eight selection instances, and 32 sample slots per instance. Actual sampling occurs in chunks of eight. First-eight probes and final-32 evaluations share the same instance/chunk seed policy; unchanged exact checkpoint evidence can be reused. This avoids inventing independent per-sample RNG seeds for a batched generation call.
- **Real Q sampling.** `prepare_round.py` samples from the separate initial Challenger adapter, using an ordinary chat prompt with an empty JSON grammar prefix. It does not import a CPU oracle program, construct the selected curricula in Python, or update Solver/Q weights during proposal generation. Every attempt is saved before JSON parsing. A complete generated JSON object plus actual EOS is required. All nine catalogue stages pass the existing frozen stage-validation gate; schema constants bind base, kind, and complete spec, and require three curricula of three stages.

## Scientific limits that remain explicit

The Solver grammar is a new sampling policy. It preserves graph structures expressible with 2–32 total nodes, including fixed START/END, all standard node types, cycles, and routes to declared nodes or NONE. Other nodes use canonical names and declaration order. These conventions preserve the represented graph family under renaming, but change token probabilities and the frequency of different graph sizes and structures. A 2,048-token cap adds a completion-length constraint. Applying the same policy to every branch controls that intervention within this pilot; it does not prove the grammar prior has no effect. Results should be described as results under this constrained policy, and should not be directly compared with the earlier unconstrained samples as though the policy were unchanged.

The Q schema is a finite nine-stage catalogue menu. Q genuinely selects stage identities and order through model sampling, but this run does not implement open proposal synthesis. The three stage interfaces remain in the engineering design; hinted-target stages are not admissible before the required hint-regret probe. No passing 512-rollout hint probe is claimed. With `n_min=8` and only three curricula in this round, no Challenger rejection-finetuning update can trigger; this does not demonstrate RFT or P5. The initial archive/context also cannot establish a long-run detour regime.

The root task reports that real job **41412533** completed one mixed binary-reward identity-stage update using the masked LoRA loss. That is actual backend update evidence, separate from the CPU controls below. This review did not rerun the GPU job or turn that stage update into a target-transfer claim; the job's saved raw samples, update record, and weights remain the authoritative evidence.

## Cost accounting and non-blocking cautions

- Current branch training is exactly `4 × 65,536 = 262,144` generated tokens if all branches complete. The plan requests 2,048 evaluation rollouts before exact-checkpoint reuse, at cap 2,048: an upper bound of 4,194,304 evaluation tokens. Q permits at most two attempts of 8,192 tokens, adding at most 16,384 Q tokens. Thus the full planned generation upper bound is **4,472,832 tokens**, excluding already completed stage screens and counting baseline within evaluation. The prepare job alone has at most 524,288 baseline plus 16,384 Q tokens. These are bounds, not observed costs.
- `Solver.generated_tokens` includes all of that Solver object's generation calls, including probes. The same name in saved `pilot_state.json` must not be read as branch training cost. The executor's phase ledgers provide the training denominator; Q attempt files and evaluation records provide their separate actual costs. Sum failed Q attempts as well as the accepted one when reporting total compute.
- Raw generation's `adapter` field identifies the loaded source adapter; after updates, it does not by itself name the current weight state. Use `optimizer_steps_before`, the branch journal, and actual checkpoint fingerprints/digests together. The reviewed executor does so.
- `Solver.save()` and `prepare_round.main()` do not independently check every output path against WORK before writing. The reviewed `RUN_PILOT.sbatch` supplies new paths beneath the actual selfgrok release, while `execute_base`/branch execution check their workspace paths. This is not a blocker for that fixed launcher; arbitrary external callers should not be treated as having the same path guarantee.
- Exact whole-worker metadata equality in `update()` is valid for the current in-process generate/update loop. It intentionally does not provide restart/resume support across a newly started worker, whose timing metadata differs. No resume capability is claimed.

## Completed CPU checks and evidence

New checks in `BACKEND_REVIEW_CPU.json`: Python AST validation for the five directly involved implementation files; all nine real catalogue entries matched their frozen stage gates and spec digests; expanded Q schema retained three branches, three items each, the fixed base, and all nine alternatives; two EOS/pad/cap trim controls passed. These checks generated no model samples or synthetic experiment outcomes.

Previously completed actual-environment CPU checks remain in `decoding/`:

- `cpu_dsl_and_json_tests.log`: 14 passing tests using the installed xgrammar, including generic DSL validity, undeclared-reference rejection, exact masks/log probabilities and gradients, EOS, JSON schema restrictions, and explicit nested-constant expansion.
- `cpu_actual_batch8.json`: actual tokenizer/worker replay for eight marked synthetic programs of different lengths; 1,013 total suffix tokens, 629 decode steps; generation masks equal replay masks for all eight rows, including differing EOS times and post-EOS padding. All fixtures parse in the actual vendor parser. These are interface fixtures, not model-generated successes.
- `cpu_actual_tokenizer_bridge.json`: training and grammar environments use matching tokenizer/vocabulary IDs; the constrained log-probability calculation has finite gradients.
- `cpu_actual_json_bridge.json`: the full actual catalogue schema compiles with 27 explicit structured-constant expansions. A marked shape fixture requires 2,064 tokens including EOS, supporting the increased 8,192 Q cap; no fixture proposal was submitted as a model proposal. The older evidence file's `mode` reflects worker mode `json_schema`; its explicit CPU/no-model flags identify its provenance.

Reproduction commands for the installed remote CPU environment, from the minimal release root:

```bash
module load python/3.11
VERGE_DECODING="$PWD/decoding"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TVM_FFI_CACHE_DIR="$VERGE_DECODING/cache/tvm_ffi" XDG_CACHE_HOME="$VERGE_DECODING/cache"
export MPLCONFIGDIR="$VERGE_DECODING/cache/mpl" TMPDIR="$VERGE_DECODING/tmp"
mkdir -p "$TVM_FFI_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
'/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python' -m unittest discover -s decoding -p 'test_*cpu.py' -v
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' decoding/cpu_batch_probe.py \
  --worker-python '/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python' \
  --model '/blue/du.j/jinjiaguo/self grok/models/Ministral-3-3B-Instruct-2512-BF16' \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output decoding/cpu_actual_batch8_NEW.json
```

The local audit used the bundled Python runtime to parse the source with `ast.parse`, call `_stage_validation` for each actual catalogue entry, expand `proposal_schema`, and assert `trim_completion([13,14,2,2,2], {2}) == [13,14,2]` and retention of a cap-truncated suffix. No implementation edits were made in this review.
