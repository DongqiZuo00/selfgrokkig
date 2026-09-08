"""CPU-only persistent xgrammar worker in the installed vLLM Python environment.

JSONL stdin/stdout RPC. Packed int32 masks are base64 encoded; no model weights,
GPU initialization, arbitrary code execution, or old VERGE rewards are involved.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from dsl_grammar import PROGRAM_PREFIX, build_grammar, grammar_digest


def encode_mask(mask):
    array = mask.numpy().astype("<i4", copy=False)
    return {"shape": list(array.shape), "dtype": "int32_le",
            "packed_base64": base64.b64encode(array.tobytes()).decode("ascii")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--json-schema", type=Path)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("TVM_FFI_CACHE_DIR", "XDG_CACHE_HOME", "TMPDIR"):
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"Explicit {name} under decoding directory required")
        resolved = Path(value).resolve()
        if not resolved.is_relative_to(Path(__file__).resolve().parent):
            raise ValueError(f"Worker writes must remain in decoding: {name}")
        resolved.mkdir(parents=True, exist_ok=True)
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, fix_mistral_regex=True)
    config = json.loads((args.model / "config.json").read_text())
    vocab_size = config.get("text_config", config)["vocab_size"]
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                            stop_token_ids=[tokenizer.eos_token_id])
    compiler = xgr.GrammarCompiler(info, max_threads=4)
    expansion_count = 0
    if args.json_schema is not None:
        from json_schema_support import expand_structured_constants
        source_schema = json.loads(args.json_schema.read_text(encoding="utf-8"))
        schema, expansion_count = expand_structured_constants(source_schema)
        compiled = compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        mode = "json_schema"
        prefix = "" if args.prefix is None else args.prefix
        grammar_sha256 = hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False,
                                                   separators=(",", ":")).encode()).hexdigest()
    else:
        compiled = compiler.compile_grammar(build_grammar(args.max_nodes))
        mode = "solver_dsl"
        prefix = PROGRAM_PREFIX if args.prefix is None else args.prefix
        grammar_sha256 = grammar_digest(args.max_nodes)
    matchers = []

    def fresh():
        matcher = xgr.GrammarMatcher(compiled)
        if prefix and not matcher.accept_string(prefix):
            raise RuntimeError("Grammar rejected the fixed format prefix")
        return matcher

    def reply(value):
        print(json.dumps(value, separators=(",", ":")), flush=True)

    metadata = {"status": "ready", "xgrammar_version": importlib.metadata.version("xgrammar"),
                "transformers_version": importlib.metadata.version("transformers"),
                "vocab_size": vocab_size, "eos_token_ids": [tokenizer.eos_token_id],
                "grammar_sha256": grammar_sha256, "max_nodes": args.max_nodes if mode == "solver_dsl" else None,
                "mode": mode, "consumed_prefix": prefix,
                "json_schema_path": str(args.json_schema) if args.json_schema else None,
                "structured_const_expansions": expansion_count,
                "tokenizer_json_sha256": hashlib.sha256((args.model / "tokenizer.json").read_bytes()).hexdigest(),
                "hf_vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True,
                                                              ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "startup_seconds": time.monotonic() - started, "cpu_only": True}
    reply(metadata)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "close":
                reply({"ok": True})
                return
            if operation == "reset":
                size = request["batch_size"]
                if type(size) is not int or not 1 <= size <= 32:
                    raise ValueError("batch size must be 1..32")
                matchers = [fresh() for _ in range(size)]
                reply({"ok": True})
            elif operation == "mask":
                if not matchers:
                    raise ValueError("reset before requesting masks")
                previous = request.get("previous_tokens")
                if previous is not None:
                    if len(previous) != len(matchers):
                        raise ValueError("one last token per matcher required")
                    for matcher, token in zip(matchers, previous):
                        if not matcher.is_terminated() and not matcher.accept_token(token):
                            raise ValueError("native generated token violated its preceding action mask")
                mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
                for index, matcher in enumerate(matchers):
                    if matcher.is_terminated():
                        mask[index].zero_()
                        token = tokenizer.eos_token_id
                        # int32 packing including high-bit stop tokens.
                        unsigned = 1 << (token % 32)
                        mask[index, token // 32] = unsigned if unsigned < 2**31 else unsigned - 2**32
                    else:
                        matcher.fill_next_token_bitmask(mask, index)
                reply({"ok": True, **encode_mask(mask)})
            elif operation == "replay":
                ids = request["token_ids"]
                if not ids or len(ids) > 8192:
                    raise ValueError("replay requires 1..8192 native suffix tokens")
                matcher = fresh()
                mask = xgr.allocate_token_bitmask(len(ids), vocab_size)
                for position, token in enumerate(ids):
                    if matcher.is_terminated():
                        raise ValueError("post-EOS padding must not enter replay or training loss")
                    matcher.fill_next_token_bitmask(mask, position)
                    if not matcher.accept_token(token):
                        raise ValueError(f"native suffix token {position} violates the recorded grammar")
                reply({"ok": True, "terminated": matcher.is_terminated(), **encode_mask(mask)})
            else:
                raise ValueError("unknown grammar service operation")
        except Exception as error:
            reply({"ok": False, "error_type": type(error).__name__, "error": str(error)})


if __name__ == "__main__":
    main()
