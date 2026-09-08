"""Actual tokenizer, eight fixed-order rows and different EOS times; CPU only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from dsl_grammar import PAINTERS, PULLERS, PROGRAM_PREFIX, example_program
from hf_grammar_bridge import GrammarService, HFGrammarLogitsProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parser", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent) or args.output.exists():
        raise ValueError("use a new result file inside decoding")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, fix_mistral_regex=True)
    kinds = PAINTERS + tuple(PULLERS)
    fixtures = [example_program(tuple(kinds[i % len(kinds)] for i in range(n)), self_loop=True)
                for n in (0, 1, 2, 3, 4, 5, 6, 30)]
    parser_verified = False
    if args.parser:
        spec = importlib.util.spec_from_file_location("verge_actual_vendor_parser_for_grammar_probe", args.parser)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for source, node_count in zip(fixtures, (2, 3, 4, 5, 6, 7, 8, 32)):
            factory = module.create_robot_factory("\n".join(source.splitlines()[1:-1]))
            if len(factory.nodes) != node_count:
                raise ValueError("actual parser sees a different node count")
            if any(route.target not in set(factory.nodes) | {"NONE"} for node in factory.nodes.values() for route in node.routes):
                raise ValueError("actual parser sees an undeclared route")
        parser_verified = True
    tokens = [tokenizer.encode(source[len(PROGRAM_PREFIX):], add_special_tokens=False) + [tokenizer.eos_token_id]
              for source in fixtures]
    prompt = tokenizer.encode(PROGRAM_PREFIX, add_special_tokens=False)
    hashes = [[] for _ in fixtures]
    with GrammarService(args.worker_python, args.model, log_name="grammar_batch_worker.stderr.log") as service:
        vocab_digest = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()
        if vocab_digest != service.metadata["hf_vocab_sha256"]:
            raise ValueError("HF training and xgrammar worker vocabulary-to-ID maps differ")
        processor = HFGrammarLogitsProcessor(service)
        native = torch.tensor([prompt] * 8)
        start = time.monotonic()
        for position in range(max(map(len, tokens))):
            scores = processor(native, torch.zeros((8, service.metadata["vocab_size"])))
            allowed = torch.isfinite(scores).cpu().numpy()
            picked = []
            for row, ids in enumerate(tokens):
                if position < len(ids):
                    token = ids[position]
                    if not allowed[row, token]:
                        raise ValueError(f"row {row} position {position} rejected")
                    packed = np.packbits(allowed[row], bitorder="little")
                    hashes[row].append(hashlib.sha256(packed.tobytes()).hexdigest())
                else:
                    # Exactly HF's treatment of rows that already sampled EOS:
                    # sampled logits are ignored, and pad is appended instead.
                    token = tokenizer.pad_token_id
                picked.append(token)
            native = torch.cat((native, torch.tensor(picked)[:, None]), dim=1)
        mask_seconds = time.monotonic() - start
        replay_started, mask_bytes = time.monotonic(), 0
        for row, ids in enumerate(tokens):
            packed, terminated = service.replay(ids)
            if not terminated:
                raise ValueError("synthetic complete DSL did not reach grammar EOS")
            replay_hashes = [hashlib.sha256(item.astype("<i4", copy=False).tobytes()).hexdigest() for item in packed]
            if hashes[row] != replay_hashes:
                raise ValueError("batch generation and training replay masks differ")
            mask_bytes += packed.nbytes
        result = {**service.metadata, "grammar_mode": service.metadata.get("mode", "solver_dsl"),
                  "mode": "CPU_fixed_fixtures_not_model_samples",
                  "batch_size": 8, "suffix_tokens_per_row_including_eos": list(map(len, tokens)),
                  "max_decode_steps": max(map(len, tokens)), "total_native_tokens": sum(map(len, tokens)),
                  "different_eos_and_post_eos_padding_checked": True,
                  "all_generation_masks_equal_training_replay": True,
                  "training_worker_vocab_maps_identical": True,
                  "actual_vendor_parser_all_fixtures_valid": parser_verified,
                  "actual_vendor_parser_path": str(args.parser) if args.parser else None,
                  "batch_rpc_mask_seconds": mask_seconds,
                  "all_replays_seconds": time.monotonic() - replay_started,
                  "packed_mask_bytes": int(mask_bytes), "gpu_or_model_weights_loaded": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
