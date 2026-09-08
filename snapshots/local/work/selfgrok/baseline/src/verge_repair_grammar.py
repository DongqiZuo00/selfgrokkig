"""CPU subprocess in the vLLM environment: reproduce its xgrammar action masks."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def build(generation_path, destination, model_path):
    os.environ["TVM_FFI_CACHE_DIR"] = str(Path(model_path).resolve().parents[1] / "caches/tvm-ffi")
    import numpy as np
    import xgrammar as xgr
    from vllm.tokenizers import get_tokenizer

    generation = json.loads(Path(generation_path).read_text())
    tokenizer = get_tokenizer(str(model_path), tokenizer_mode="hf")
    # The inference environment intentionally has a different Transformers
    # version. Vocabulary size is static config data, not a model-class load.
    config = json.loads((Path(model_path) / "config.json").read_text())
    vocab_size = config.get("text_config", config)["vocab_size"]
    # Exactly vLLM 0.21 backend_xgrammar's non-MistralTokenizer path. The server
    # is explicitly --tokenizer-mode hf, backend xgrammar, any_whitespace=True.
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info, max_threads=8)
    ctx = compiler.compile_json_schema(json.dumps(generation["schema"]), any_whitespace=True)
    arrays = {"vocab_size": np.array(vocab_size),
              "grammar_version": np.array(importlib.metadata.version("xgrammar"))}
    for candidate in generation["candidates"]:
        ids = candidate["completion_token_ids"]
        matcher = xgr.GrammarMatcher(ctx)
        packed = xgr.allocate_token_bitmask(len(ids), vocab_size)
        for pos, token in enumerate(ids):
            matcher.fill_next_token_bitmask(packed, pos)
            if not matcher.accept_token(token):
                raise RuntimeError(f"Sampled action rejected by server-equivalent grammar: {candidate['index']}:{pos}")
        arrays[f"candidate_{candidate['index']}"] = packed.numpy()
        arrays[f"tokens_{candidate['index']}"] = np.array(ids, dtype=np.int64)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    print(json.dumps({"grammar_masks_ready": str(path), "vocab_size": vocab_size,
        "tokens": sum(len(c["completion_token_ids"]) for c in generation["candidates"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generation")
    parser.add_argument("destination")
    parser.add_argument("model")
    args = parser.parse_args()
    build(args.generation, args.destination, args.model)
