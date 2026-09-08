"""Canonical Manufactoria graph grammar; no target, oracle, or reward definition.

Two through 32 nodes, including START start and END end. Every intermediate
graph can be renamed n0..nK without changing its executable behavior. All six
standard intermediate node types, all declared destinations, and cycles remain.
"""
import hashlib
import json

PROGRAM_PREFIX = "```manufactoria\nSTART start:\n    NEXT "
PAINTERS = ("PAINTER_RED", "PAINTER_BLUE", "PAINTER_YELLOW", "PAINTER_GREEN")
PULLERS = {"PULLER_RB": ("R", "B", "EMPTY"), "PULLER_YG": ("Y", "G", "EMPTY")}


def literal(text):
    return json.dumps(text, ensure_ascii=True)


def build_grammar(max_nodes=32):
    if type(max_nodes) is not int or not 2 <= max_nodes <= 32:
        raise ValueError("max_nodes includes mandatory start/end and must be 2..32")
    rules = ["root ::= " + " | ".join(f"program{k}" for k in range(max_nodes - 1))]
    rules.append("painter ::= " + " | ".join(map(literal, PAINTERS)))
    for k in range(max_nodes - 1):
        references = ("start", "end", "NONE") + tuple(f"n{i}" for i in range(k))
        rules.append(f"ref{k} ::= " + " | ".join(map(literal, references)))
        program = [literal(PROGRAM_PREFIX), f"ref{k}", literal("\n")]
        program.extend(f"node{k}_{i}" for i in range(k))
        program.append(literal("END end\n```"))
        rules.append(f"program{k} ::= " + " ".join(program))
        for i in range(k):
            alternatives = [f'painter {literal(f" n{i}:" + chr(10) + "    NEXT ")} ref{k} {literal(chr(10))}']
            for node_type, conditions in PULLERS.items():
                parts = [literal(f"{node_type} n{i}:\n")]
                for condition in conditions:
                    parts.extend((literal(f"    [{condition}] "), f"ref{k}", literal("\n")))
                alternatives.append(" ".join(parts))
            rules.append(f"node{k}_{i} ::= " + " | ".join(alternatives))
    return "\n".join(rules) + "\n"


def grammar_digest(max_nodes=32):
    return hashlib.sha256(build_grammar(max_nodes).encode()).hexdigest()


def example_program(node_types=(), *, self_loop=False):
    """Synthetic grammar fixture. It is not an oracle and performs no target task."""
    types = tuple(node_types)
    if len(types) > 30 or any(kind not in PAINTERS + tuple(PULLERS) for kind in types):
        raise ValueError("unknown type or too many nodes")
    lines = [PROGRAM_PREFIX + ("n0" if types else "end")]
    for index, kind in enumerate(types):
        destination = f"n{index}" if self_loop else (f"n{index+1}" if index+1 < len(types) else "end")
        lines.append(f"{kind} n{index}:")
        if kind in PAINTERS:
            lines.append(f"    NEXT {destination}")
        else:
            lines.extend(f"    [{condition}] {destination}" for condition in PULLERS[kind])
    lines.extend(("END end", "```"))
    return "\n".join(lines)
