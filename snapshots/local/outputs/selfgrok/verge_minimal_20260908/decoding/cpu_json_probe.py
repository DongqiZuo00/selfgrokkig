"""CPU test of actual caller catalogue/schema, never a Challenger model proposal.

An explicitly synthetic value is used solely to test token masks. It is retained
only as a SHA/length, not written into any runnable proposal or experiment input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from hf_grammar_bridge import GrammarService


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--worker-python", required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--catalogue", type=Path, required=True)
    cli.add_argument("--schema", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent) or args.output.exists():
        raise ValueError("use a new CPU report under decoding")
    schema = json.loads(args.schema.read_text())
    base = schema["properties"]["base_checkpoint"]["const"]
    catalogue = json.loads(args.catalogue.read_text())["stages"]
    entry = {key: catalogue[0][key] for key in ("stage_id", "kind", "spec")}
    fixture = {"base_checkpoint": base, "curricula": {g: [entry] * 3 for g in ("g1", "g2", "g3")}}
    fixture_text = json.dumps(fixture, separators=(",", ":"), ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, fix_mistral_regex=True)
    token_ids = tokenizer.encode(fixture_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    with GrammarService(args.worker_python, args.model, json_schema_path=args.schema,
                        prefix="", log_name="json_worker.stderr.log") as service:
        started = time.monotonic()
        masks, terminated = service.replay(token_ids)
        if not terminated:
            raise ValueError("complete synthetic catalogue-shaped JSON did not reach EOS")
        report = {**service.metadata, "grammar_mode": service.metadata["mode"],
                  "mode": "CPU_JSON_SCHEMA_FIXTURE_NOT_MODEL_PROPOSAL",
                  "fixture_json_sha256": hashlib.sha256(fixture_text.encode()).hexdigest(),
                  "fixture_native_tokens_including_eos": len(token_ids), "json_fixture_characters": len(fixture_text),
                  "all_native_actions_accepted": True, "grammar_terminated": terminated,
                  "replay_seconds": time.monotonic() - started, "mask_bytes": int(masks.nbytes),
                  "proposal_written_or_submitted": False, "gpu_or_model_weights_loaded": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
