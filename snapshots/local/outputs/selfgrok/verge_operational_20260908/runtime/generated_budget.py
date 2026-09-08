"""Generated-token accounting for VERGE operational v1 (no model dependency).

One complete eight-rollout group comes from one prompt. At a phase boundary,
remaining < 8 tokens are sampled without learning, explicitly logged as drain.
This keeps every complete rollout intact and matches actual generation, not
loss-mask positions. Prompt, probe, Challenger and wall-clock costs are separate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Rollout:
    prompt_id: str
    task_kind: str
    completion_token_ids: tuple[int, ...]
    reward: int
    raw_completion: str
    verifier_digest: str

    def validate(self, cap: int) -> None:
        if not self.prompt_id or not self.verifier_digest:
            raise ValueError("Each rollout needs prompt and frozen verifier identity")
        if self.task_kind not in {"target", "stage"}:
            raise ValueError("Task kind must identify the binary verifier used")
        if type(self.reward) is not int or self.reward not in (0, 1):
            raise ValueError("Solver reward is binary full pass only")
        if not 1 <= len(self.completion_token_ids) <= cap:
            raise ValueError("A rollout must make progress within the requested cap")
        if any(type(t) is not int or t < 0 for t in self.completion_token_ids):
            raise ValueError("Invalid completion token ids")


def binary_advantages(rewards: Sequence[int]) -> tuple[float, ...]:
    if len(rewards) != 8 or any(type(r) is not int or r not in (0, 1) for r in rewards):
        raise ValueError("Expected a complete eight-rollout binary reward group")
    p = sum(rewards) / 8
    if p in (0, 1):
        return (0.0,) * 8
    scale = math.sqrt(p * (1 - p)) + 1e-6
    return tuple((r - p) / scale for r in rewards)


class JsonlJournal:
    """Append-only event journal with a local SHA chain; never truncate old logs."""

    def __init__(self, path: Path, run_id: str):
        if path.exists():
            raise FileExistsError("Use a new run directory; journal resume is explicit")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.run_id, self.previous, self.sequence = path, run_id, "0" * 64, 0

    def append(self, kind: str, payload: dict) -> None:
        record = {"run_id": self.run_id, "sequence": self.sequence,
                  "previous_sha256": self.previous, "kind": kind, "payload": payload}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        record["sha256"] = digest
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        self.previous, self.sequence = digest, self.sequence + 1


def verify_journal(path: Path) -> int:
    previous = "0" * 64
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, record in enumerate(records):
        digest = record.pop("sha256")
        if record["previous_sha256"] != previous or record["sequence"] != index:
            raise ValueError("Broken journal ordering")
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(encoded.encode()).hexdigest() != digest:
            raise ValueError("Journal content was changed")
        previous = digest
    return len(records)


@dataclass
class PhaseLedger:
    name: str
    quota: int
    generated_tokens: int = 0
    loss_eligible_tokens: int = 0
    nonzero_advantage_tokens: int = 0
    boundary_drain_tokens: int = 0
    rollouts: int = 0
    optimizer_steps: int = 0
    constant_groups: int = 0

    def __post_init__(self):
        _positive_int(self.quota, "quota")

    @property
    def remaining(self) -> int:
        return self.quota - self.generated_tokens


def run_phase(ledger: PhaseLedger, *, generate: Callable[[int, int, bool], Sequence[Rollout]],
              update: Callable[[Sequence[Rollout], Sequence[float]], None],
              journal: JsonlJournal, max_new_tokens: int = 2048) -> PhaseLedger:
    """generate(n, cap, drain): draw n independent rollouts of the same prompt.

    The backend chooses target/stage prompt using the frozen mixture. A drain
    uses that same phase distribution but never updates. No verifier partial
    scores are accepted by this interface. Nonconstant groups call update once;
    constant groups do not call optimizer/scheduler/model training at all.
    """
    _positive_int(max_new_tokens, "max_new_tokens")
    if any(value != 0 for key, value in asdict(ledger).items() if key not in {"name", "quota"}):
        raise ValueError("run_phase requires a fresh ledger; resume needs journal reconciliation")
    journal.append("phase_start", asdict(ledger))
    while ledger.remaining:
        drain = ledger.remaining < 8
        count = 1 if drain else 8
        cap = min(max_new_tokens, ledger.remaining // count)
        journal.append("generation_attempt", {"phase": ledger.name, "count": count, "cap": cap,
                                              "remaining": ledger.remaining, "drain": drain})
        outputs = None
        try:
            outputs = tuple(generate(count, cap, drain))
            if len(outputs) != count:
                raise ValueError("Backend returned an incomplete reward group")
            for output in outputs:
                output.validate(cap)
            if len({(r.prompt_id, r.task_kind, r.verifier_digest) for r in outputs}) != 1:
                raise ValueError("Reward group mixes prompts or verifier versions")
        except Exception as error:
            records = [asdict(r) if isinstance(r, Rollout) else repr(r) for r in outputs] if outputs is not None else None
            journal.append("rejected_generation", {"phase": ledger.name, "error": repr(error),
                           "records": records,
                           "known_generated_tokens": sum(len(r.completion_token_ids) for r in outputs)
                               if outputs is not None and all(isinstance(r, Rollout) for r in outputs) else None,
                           "budget_complete": False})
            raise
        tokens = sum(len(r.completion_token_ids) for r in outputs)
        ledger.generated_tokens += tokens
        ledger.rollouts += count
        # Persist generated work before update: an update error remains auditable.
        journal.append("rollouts", {"phase": ledger.name, "drain": drain,
                       "cap": cap, "records": [asdict(r) for r in outputs]})
        if drain:
            ledger.boundary_drain_tokens += tokens
        else:
            ledger.loss_eligible_tokens += tokens
            advantages = binary_advantages([r.reward for r in outputs])
            if any(advantages):
                update(outputs, advantages)
                ledger.optimizer_steps += 1
                ledger.nonzero_advantage_tokens += tokens
            else:
                ledger.constant_groups += 1
        journal.append("accounting", asdict(ledger))
    if ledger.generated_tokens != ledger.quota or ledger.boundary_drain_tokens > 7:
        raise AssertionError("Generated-token budget mismatch")
    journal.append("phase_complete", asdict(ledger))
    return ledger


def assert_fresh_optimizer(optimizer, *, beta: float, weight_decay: float) -> None:
    if beta != 0 or weight_decay != 0:
        raise ValueError("Operational Solver requires beta=0 and weight_decay=0")
    if optimizer.state:
        raise ValueError("Branch optimizer must be fresh at the shared base")
    if any(group.get("weight_decay", 0) != 0 for group in optimizer.param_groups):
        raise ValueError("Actual optimizer has nonzero weight decay")
