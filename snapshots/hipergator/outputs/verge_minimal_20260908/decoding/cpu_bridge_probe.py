"""Actual installed-tokenizer CPU integration probe, no weights or GPU loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from dsl_grammar import PROGRAM_PREFIX, example_program
from hf_grammar_bridge import GrammarService, HFGrammarLogitsProcessor, masked_chosen_logp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("CPU probe outputs stay inside decoding")
    if args.output.exists():
        raise FileExistsError("preserve prior CPU probe results")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, fix_mistral_regex=True)
    source = example_program(("PULLER_RB", "PAINTER_GREEN"), self_loop=True)
    suffix = source[len(PROGRAM_PREFIX):]
    ids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    prompt_ids = tokenizer.encode(PROGRAM_PREFIX, add_special_tokens=False)
    with GrammarService(args.worker_python, args.model) as service:
        if service.metadata["tokenizer_json_sha256"] != hashlib.sha256((args.model / "tokenizer.json").read_bytes()).hexdigest():
            raise ValueError("training and worker tokenizers read different frozen files")
        start = time.monotonic()
        processor = HFGrammarLogitsProcessor(service)
        native = torch.tensor([prompt_ids], dtype=torch.long)
        generation_allowed = []
        for token in ids:
            scores = processor(native, torch.zeros((1, service.metadata["vocab_size"])))
            allowed = torch.isfinite(scores).cpu().numpy()
            if not allowed[0, token]:
                raise ValueError(f"actual tokenizer's native suffix token {token} was excluded")
            generation_allowed.append(allowed[0])
            native = torch.cat((native, torch.tensor([[token]])), dim=1)
        mask_seconds = time.monotonic() - start
        replay_started = time.monotonic()
        packed, terminated = service.replay(ids)
        replay_seconds = time.monotonic() - replay_started
        if not terminated:
            raise ValueError("complete synthetic fixture did not terminate")
        unpacked = np.unpackbits(packed.view(np.uint8), axis=-1, bitorder="little")[:, :service.metadata["vocab_size"]]
        np.testing.assert_array_equal(np.array(generation_allowed), unpacked.astype(bool))
        logits = torch.zeros((len(ids), service.metadata["vocab_size"]), requires_grad=True)
        logp = masked_chosen_logp(logits, torch.tensor(ids), packed)
        expected = -np.log(unpacked.sum(-1))
        np.testing.assert_allclose(logp.detach().numpy(), expected, rtol=1e-6, atol=1e-6)
        (-logp.mean()).backward()
        if not torch.isfinite(logits.grad).all():
            raise ValueError("masked loss produced nonfinite gradients")
        report = {**service.metadata, "grammar_mode": service.metadata.get("mode", "solver_dsl"),
                  "mode": "CPU_tokenizer_and_action_mask_fixture_not_model_samples",
                  "fixture_suffix_tokens_including_eos": len(ids), "prefix_loss_tokens": 0,
                  "generation_replay_masks_exactly_equal": True, "masked_logp_matches_sampling": True,
                  "autograd_finite": True, "one_row_rpc_masks_seconds": mask_seconds,
                  "replay_seconds": replay_seconds, "packed_mask_bytes": int(packed.nbytes),
                  "actual_hf_environment": "caller; no vLLM site-packages injected",
                  "gpu_or_model_weights_loaded": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
