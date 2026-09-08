"""Independent numeric spelling variant of the frozen canonical DSL grammar.

Only the terminal identifiers n0..n29 become 0..29. This does not rewrite a
sampled completion, restrict graph topology, or supply a program solution.
"""
import hashlib
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsl_grammar import PROGRAM_PREFIX, PAINTERS, PULLERS
from dsl_grammar import build_grammar as _frozen_build, grammar_digest as _frozen_digest

POLICY_ID = "dsl_grammar_numeric_v1"


def rename_identifiers(text):
    """For grammar construction and CPU fixtures ONLY; never a sample repair."""
    return re.sub(r"\bn([0-9]+)\b", lambda match: match[1], text)


def build_grammar(max_nodes=32):
    # Frozen rule names are nodeK_I/programK/refK, so this substitution changes
    # terminal node names only. Bounds and all graph choices stay untouched.
    return rename_identifiers(_frozen_build(max_nodes))


def grammar_digest(max_nodes=32):
    return hashlib.sha256(build_grammar(max_nodes).encode()).hexdigest()


def parent_grammar_digest(max_nodes=32):
    return _frozen_digest(max_nodes)
