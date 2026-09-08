"""HF grammar masking and matching masked-policy training, without mixed envs.

The training process keeps its own Torch/Transformers. A persistent subprocess
uses the already-installed xgrammar in the vLLM environment on CPU only.
"""
import base64
import json
import os
from pathlib import Path
import subprocess
import time


def decode_mask(response):
    import numpy as np
    if response.get("dtype") != "int32_le":
        raise ValueError("unknown action mask encoding")
    return np.frombuffer(base64.b64decode(response["packed_base64"], validate=True), dtype="<i4").reshape(response["shape"]).copy()


class GrammarService:
    def __init__(self, python_executable, model_path, *, max_nodes=32, log_name="grammar_worker.stderr.log",
                 json_schema_path=None, prefix=None):
        directory = Path(__file__).resolve().parent
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           TVM_FFI_CACHE_DIR=str(directory / "cache/tvm_ffi"),
                           XDG_CACHE_HOME=str(directory / "cache"), TMPDIR=str(directory / "tmp"))
        for path in (directory / "cache/tvm_ffi", directory / "tmp"):
            path.mkdir(parents=True, exist_ok=True)
        log_path = (directory / log_name).resolve()
        if not log_path.is_relative_to(directory):
            raise ValueError("worker log must stay in decoding")
        self.stderr = log_path.open("ab")
        started = time.monotonic()
        command = [str(python_executable), "-u", str(directory / "grammar_service.py"),
                   "--model", str(model_path), "--max-nodes", str(max_nodes)]
        if json_schema_path is not None:
            command.extend(("--json-schema", str(json_schema_path)))
        if prefix is not None:
            command.extend(("--prefix", prefix))
        self.process = subprocess.Popen(command,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                        env=environment, cwd=str(directory), text=True, encoding="utf-8")
        self.metadata = self._read()
        self.metadata["client_worker_startup_seconds"] = time.monotonic() - started
        if self.metadata.get("status") != "ready":
            raise RuntimeError("grammar worker did not initialize")

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"grammar worker stopped (returncode={self.process.poll()}); inspect its stderr log")
        result = json.loads(line)
        if result.get("ok") is False:
            raise ValueError("grammar worker: " + result.get("error", "unknown failure"))
        return result

    def request(self, operation, **arguments):
        self.process.stdin.write(json.dumps({"op": operation, **arguments}) + "\n")
        self.process.stdin.flush()
        return self._read()

    def reset(self, batch_size):
        self.request("reset", batch_size=batch_size)

    def next_mask(self, previous_tokens=None):
        return decode_mask(self.request("mask", previous_tokens=previous_tokens))

    def replay(self, completion_token_ids):
        response = self.request("replay", token_ids=list(completion_token_ids))
        return decode_mask(response), response["terminated"]

    def close(self):
        if self.process.poll() is None:
            self.request("close")
            self.process.wait(timeout=10)
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def allowed_tensor(packed, vocab_size, device):
    import torch
    bits = torch.as_tensor(packed, dtype=torch.int32, device=device).long()
    if bits.ndim != 2 or bits.shape[-1] * 32 < vocab_size:
        raise ValueError("mask must cover the full model output vocabulary")
    ids = torch.arange(vocab_size, device=device)
    return ((bits[:, ids // 32] >> (ids % 32)) & 1).bool()


class HFGrammarLogitsProcessor:
    """One instance per generate() call; batch identity and one-token steps fixed.

    Inputs include service.metadata['consumed_prefix'] as prompt context. This is
    PROGRAM_PREFIX in DSL mode and usually '' (ordinary chat prompt) in JSON mode.
    Greedy/beam/speculative/reordered batching is not supported by this adapter.
    """
    def __init__(self, service):
        self.service, self.width, self.batch_size = service, None, None

    def __call__(self, input_ids, scores):
        import torch
        size, width = input_ids.shape
        if self.width is None:
            self.batch_size = size
            self.service.reset(size)
            previous = None
        else:
            if size != self.batch_size or width != self.width + 1:
                raise ValueError("grammar processor requires fixed-order sampling with one new token per call")
            previous = input_ids[:, -1].detach().cpu().tolist()
        if scores.shape != (size, self.service.metadata["vocab_size"]):
            raise ValueError("logits width differs from grammar's full model vocabulary")
        packed = self.service.next_mask(previous)
        allowed = allowed_tensor(packed, scores.shape[-1], scores.device)
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("grammar produced an empty allowed action set")
        self.width = width
        return scores.masked_fill(~allowed, -torch.inf)


def native_sampling_kwargs():
    """No extra probability warpers: loss below normalizes this exact policy."""
    return dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, top_h=None, typical_p=1.0,
                min_p=None, repetition_penalty=1.0, no_repeat_ngram_size=0,
                num_beams=1, num_return_sequences=1, renormalize_logits=False,
                bad_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
                forced_bos_token_id=None, forced_eos_token_id=None, sequence_bias=None,
                stop_strings=None, epsilon_cutoff=0.0, eta_cutoff=0.0,
                encoder_repetition_penalty=1.0, exponential_decay_length_penalty=None,
                min_length=0, min_new_tokens=0, watermarking_config=None, guidance_scale=None)


def masked_chosen_logp(logits, targets, packed, *, chunk=32):
    """Autograd-preserving log pi_grammar(a|context), same mask as generation.

    Packed rows come from replay of the ACTUAL native suffix IDs, including its
    generated EOS and excluding batch padding. Prefix bytes are context only.
    """
    import torch
    if logits.ndim != 2 or len(targets) != logits.shape[0] or len(packed) != len(targets):
        raise ValueError("one action mask and target per native generated token required")
    pieces = []
    for first in range(0, len(targets), chunk):
        part = logits[first:first+chunk].float()
        chosen = targets[first:first+chunk]
        allowed = allowed_tensor(packed[first:first+chunk], part.shape[-1], part.device)
        if not bool(allowed.gather(1, chosen[:, None]).all()):
            raise ValueError("observed action is outside the SAME grammar policy used for generation")
        restricted = part.masked_fill(~allowed, -torch.inf)
        pieces.append(restricted.gather(1, chosen[:, None]).squeeze(-1) - torch.logsumexp(restricted, dim=-1))
    return torch.cat(pieces)


def masked_suffix_loss(model, prompt_token_ids, completion_token_ids, advantage, packed):
    """Sum of -binary-group advantage * constrained-policy logp, no partial reward."""
    import torch
    if not prompt_token_ids or not completion_token_ids:
        raise ValueError("nonempty prompt and actual native suffix required")
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_token_ids) + list(completion_token_ids)], dtype=torch.long, device=device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = result.logits[0, len(prompt_token_ids)-1:-1]
    targets = ids[0, len(prompt_token_ids):]
    logp = masked_chosen_logp(logits, targets, packed)
    return (-float(advantage) * logp).sum(), logp
