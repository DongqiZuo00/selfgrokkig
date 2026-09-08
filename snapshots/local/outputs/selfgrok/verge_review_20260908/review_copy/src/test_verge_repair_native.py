"""CPU-only compilation of the real vLLM schema, with no model loading or sampling."""
import json
import importlib.metadata
import xgrammar
from verge_round_core import config
from verge_repair_protocol import proposal_schema

schema = proposal_schema(config()["families"])
grammar = xgrammar.Grammar.from_json_schema(json.dumps(schema))
assert grammar is not None
print(json.dumps({"schema_compiles": True, "vllm": importlib.metadata.version("vllm"),
    "xgrammar": importlib.metadata.version("xgrammar"), "model_sampling": False,
    "warning": "Schema compilation does not validate end-to-end learning or masked policy-gradient integration"}))
