"""Explicit state machine, hypothesis ledger, budgets and the append-only JSONL trace."""
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class State(str, enum.Enum):
    INITIAL = "INITIAL"
    REPRODUCING = "REPRODUCING"
    INVESTIGATING = "INVESTIGATING"
    HYPOTHESIZING = "HYPOTHESIZING"
    EXPERIMENTING = "EXPERIMENTING"
    PATCHING = "PATCHING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    DONE = "DONE"
    ABSTAINED = "ABSTAINED"              # honest "I could not do this"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


TERMINAL = {State.DONE, State.ABSTAINED, State.BUDGET_EXHAUSTED}

_S = State
_ALLOWED: dict[State, set[State]] = {
    _S.INITIAL: {_S.REPRODUCING},
    _S.REPRODUCING: {_S.INVESTIGATING, _S.ABSTAINED},
    _S.INVESTIGATING: {_S.HYPOTHESIZING, _S.PATCHING, _S.ABSTAINED},   # PATCHING directly = baseline mode
    _S.HYPOTHESIZING: {_S.HYPOTHESIZING, _S.EXPERIMENTING, _S.PATCHING, _S.ABSTAINED},  # PATCHING: baseline mode only
    _S.EXPERIMENTING: {_S.PATCHING, _S.HYPOTHESIZING, _S.ABSTAINED},
    _S.PATCHING: {_S.VERIFYING, _S.HYPOTHESIZING, _S.ABSTAINED},
    _S.VERIFYING: {_S.REVIEWING, _S.HYPOTHESIZING, _S.ABSTAINED},
    _S.REVIEWING: {_S.DONE, _S.HYPOTHESIZING, _S.ABSTAINED},
}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self) -> None:
        self.state = State.INITIAL
        self.history: list[State] = [State.INITIAL]

    def to(self, new: State) -> None:
        if self.state in TERMINAL:
            raise IllegalTransition(f"{self.state.value} is terminal")
        if new is State.BUDGET_EXHAUSTED or new in _ALLOWED.get(self.state, set()):
            self.state = new
            self.history.append(new)
            return
        raise IllegalTransition(f"{self.state.value} -> {new.value} is not allowed")


@dataclass
class Hypothesis:
    id: str
    claim: str
    prediction: str
    experiment: dict[str, Any]
    status: str = "untested"             # untested | supported | refuted | inconclusive
    reason: str = ""
    evidence_events: list[int] = field(default_factory=list)
    iteration: int = 0


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_llm_calls: int = 60
    max_seconds: float = 1800.0
    max_iterations: int = 3
    llm_calls: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def charge(self, n: int = 1) -> None:
        self.llm_calls += n

    def check(self) -> None:
        if self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.max_llm_calls})")
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded(f"time budget exhausted ({self.max_seconds:.0f}s)")


class Trace:
    """Append-only JSONL event log. Every claim in the final report points at an event id here."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.records: list[dict[str, Any]] = []
        self.state = State.INITIAL.value

    def emit(self, kind: str, **data: Any) -> int:
        rec = {"id": len(self.records) + 1, "t": round(time.time(), 3),
               "state": self.state, "kind": kind, "data": data}
        self.records.append(rec)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        return rec["id"]

    def get(self, event_id: int) -> dict[str, Any] | None:
        return self.records[event_id - 1] if 0 < event_id <= len(self.records) else None

    def close(self) -> None:
        self._fh.close()

    @staticmethod
    def load(path: Path | str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
