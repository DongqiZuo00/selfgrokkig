"""Numeric grammar worker with the existing reset/mask/replay RPC interface.

The frozen logits processor and masked loss can consume this object unchanged.
The caller MUST record and check metadata['decoding_policy']; the legacy
backend's hard-coded dsl_grammar_v1 label is not correct for this new policy.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_grammar_bridge import GrammarService as _ExistingClient
from numeric_grammar import POLICY_ID, grammar_digest


class NumericGrammarService(_ExistingClient):
    def __init__(self, python_executable, model_path, *, max_nodes=32,
                 log_name="numeric_worker.stderr.log"):
        expected_digest = grammar_digest(max_nodes)
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
            raise ValueError("numeric worker log must stay in numeric_variant")
        self.stderr = log_path.open("ab")
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                [str(python_executable), "-u", str(directory / "numeric_service.py"),
                 "--model", str(model_path), "--max-nodes", str(max_nodes)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                env=environment, cwd=str(directory), text=True, encoding="utf-8")
            self.metadata = self._read()
            self.metadata["client_worker_startup_seconds"] = time.monotonic() - started
            if (self.metadata.get("status") != "ready"
                    or self.metadata.get("decoding_policy") != POLICY_ID
                    or self.metadata.get("grammar_sha256") != expected_digest):
                raise RuntimeError("numeric worker identity does not match its client")
        except BaseException:
            if getattr(self, "process", None) is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.stderr.close()
            raise
