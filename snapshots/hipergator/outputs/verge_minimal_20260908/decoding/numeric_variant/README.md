# Numeric canonical-name CPU backup

This is an independent, unselected backup while names-only prompt probe 41415498 runs. No GPU job or model sampling is launched by this directory. A useful result from that prompt probe can make this backup unnecessary.

The sole grammar change is terminal node spelling: `n0…n29` becomes `0…29`. START `start`, END `end`, 2–32 total nodes, all six intermediate node types, declared references, forward/backward routes, self-loops, other cycles, and NONE remain as in v1. There is no required START destination or task-specific topology. Numeric naming remains canonical and contiguous; it is compatible with the generic unique-identifier description, but does not accept every possible descriptive identifier.

`numeric_grammar.py` applies a word-boundary replacement to the grammar's terminal names. Existing nonterminal names (`programK`, `refK`, `nodeK_I`) are unaffected. Its `rename_identifiers()` helper is used only to construct the grammar and CPU fixtures. It must never rewrite actual model completions before verification or training.

The motivation is recorded in `../PROMPT_ALIGNMENT_REVIEW.md`: 80/96 unconstrained stage samples began with actual token 1048=`0` or 1049=`1`, which the n-name grammar excludes at the first suffix position. Admitting these tokens is an engineering hypothesis, not an observed improvement. Graph equivalence under renaming does not imply the same sampling prior, token lengths, or model success rates.

## Files and unchanged sources

- `numeric_grammar.py`: numeric grammar construction and its own SHA.
- `numeric_service.py`: independent worker using the prior wire protocol, numeric compiler, explicit `mode=solver_dsl_numeric` and `decoding_policy=dsl_grammar_numeric_v1`. Only the fixed DSL prefix is accepted; JSON mode is not exposed by this backup.
- `numeric_bridge.py`: `NumericGrammarService`, with inherited `reset`, `next_mask`, `replay`, `request`, `close` and context-manager methods. Worker identity/digest is checked on startup. All worker caches and logs stay under `numeric_variant`.
- `test_numeric_cpu.py`, `cpu_numeric_probe.py`: CPU fixtures and actual-environment checks, never generated curricula, oracle solutions, or training data.
- `FROZEN_SOURCE_GUARDS.json`: hashes of the unchanged parent `dsl_grammar.py`, `grammar_service.py`, and `hf_grammar_bridge.py`; the CPU tests verify them.

## Backend integration must explicitly name the new policy

The service is compatible with the existing `HFGrammarLogitsProcessor` and `masked_suffix_loss`. One service is reused per Solver; each generation call still needs a fresh logits processor. Replay consumes the actual native numeric suffix IDs, including its EOS and excluding later batch padding. No separate token remapping or unmasked loss is permitted.

```python
import sys
sys.path.insert(0, str(minimal_release_root / 'decoding/numeric_variant'))
from numeric_bridge import NumericGrammarService
from hf_grammar_bridge import HFGrammarLogitsProcessor, native_sampling_kwargs

# Interface example only; no GPU invocation is performed by this document.
grammar = NumericGrammarService(vllm_python, actual_model_directory)
assert grammar.metadata['decoding_policy'] == 'dsl_grammar_numeric_v1'
# A future Solver may hold this object as solver.grammar.
# Its native generation processor and masked loss use this same object.
```

**Do not simply swap the service into the frozen backend and keep its recorded label.** The reviewed `hf_backend.py` hard-codes `dsl_grammar_v1` in both raw generation and the update check. `execute_round.py` also freezes that label in its baseline contract and checks it for evaluations. Although replay would then use the numeric masks, the saved policy label would be false. A future caller must explicitly record and verify `dsl_grammar_numeric_v1`, retain the numeric grammar SHA, and use a new baseline contract/evidence under this same policy. Its sample metadata must match the worker used for loss. No backend, existing baseline, or frozen manifest is changed here.

The unchanged Solver controls still apply: actual binary full pass only, complete eight-sample reward groups, skip constant groups without optimizer/scheduler mutation, fresh optimizer per shared-base branch, dropout/KL beta/weight decay zero, and generated-token accounting including EOS. Numeric spelling does not change these controls.

## CPU verification

The unit checks cover bounds, all node types, arbitrary allowed route destinations including loops, undeclared/duplicate/bad spellings rejected, first-position numeric actions admitted and premature EOS rejected, and unchanged frozen source hashes.

The actual-tokenizer probe checks all 31 total-node counts from 2 through 32 under linear, self-loop and mixed-route synthetic fixtures: 93 graph pairs, with actual vendor-parser node/type/edge signatures matched under renaming. Each pair is executed on seven generic tape fixtures, giving 651 paired executions including termination, output, rejection reason, and renamed execution path. These computations do not evaluate a benchmark model solution.

The same probe exercises eight different EOS times using the installed HF tokenizer and persistent xgrammar CPU worker; it compares every generation mask against training replay. It also tests post-EOS pad rejection, cap-truncated prefix replay, exact masked log probabilities, finite autograd, and zero gradient on excluded actions.

**Completed: all six actual-xgrammar CPU tests passed, and every actual-tokenizer/vendor-parser probe check passed.** Results are in `cpu_numeric_actual_v1.json`, `cpu_numeric_unit_v1.log`, and `cpu_numeric_probe_v1.log`. The numeric grammar SHA is `578295b23cde3f0f8d31e103ba9bf3bd8a2f28d9e1aed20cc150ec1b49ab1bcc`; its parent grammar SHA remains `46ee784f8dad729fccfe290dd80560a5f14f7b822acffee2df1467f752ef9e5e`.

The eight fixture suffix lengths, including EOS, were `[7,20,34,47,60,90,120,628]`, totaling 1,006 tokens. Worker cold startup took 15.47 seconds; 628 batch mask steps took 7.00 seconds; replay took 0.20 seconds. Packed masks totaled 16,482,304 bytes. These are CPU interface costs, not GPU throughput measurements. The actual initial mask admits tokens 1048/1049 (`0`/`1`), excludes 1110 (`n`), retains `start`/`end`, and still excludes premature EOS. Nothing here demonstrates a positive numeric-policy model reward.

Reproduce from this remote `numeric_variant` directory with new output filenames:

```bash
module load python/3.11
VERGE_NUMERIC="$PWD"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TVM_FFI_CACHE_DIR="$VERGE_NUMERIC/cache/tvm_ffi" XDG_CACHE_HOME="$VERGE_NUMERIC/cache"
export MPLCONFIGDIR="$VERGE_NUMERIC/cache/mpl" TMPDIR="$VERGE_NUMERIC/tmp" HF_HOME="$VERGE_NUMERIC/cache/huggingface"
mkdir -p "$TVM_FFI_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR"
'/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python' -B -m unittest -v test_numeric_cpu
'/blue/du.j/jinjiaguo/self grok/envs/uncertainty/bin/python' -B cpu_numeric_probe.py \
  --worker-python '/blue/du.j/jinjiaguo/self grok/envs/vllm/bin/python' \
  --model '/blue/du.j/jinjiaguo/self grok/models/Ministral-3-3B-Instruct-2512-BF16' \
  --parser '/blue/du.j/jinjiaguo/self grok/vendor/rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py' \
  --output "$VERGE_NUMERIC/cpu_numeric_actual_NEW.json"
```

There is no numeric-policy GPU evidence, no numeric model success rate, and no decision to adopt this variant. The root task retains that decision after the names-only prompt probe.
