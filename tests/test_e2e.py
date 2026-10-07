"""End-to-end runs of the whole state machine with a scripted LLM (no model, no network)."""
import json
from pathlib import Path

import pytest

from autodebug.agent import Config, Debugger
from autodebug.evaluate import holdout_passes
from autodebug.llm import ScriptedLLM
from autodebug.state import State, Trace

CASES = Path(__file__).resolve().parents[1] / "cases"

REPRO = '''import threading
from registry.service import UserRegistry, DuplicateUser

def test_concurrent_duplicate_registration_is_rejected_cleanly():
    reg = UserRegistry()
    n = 8
    barrier = threading.Barrier(n)
    results = []
    def worker():
        barrier.wait()
        try:
            reg.register("alice", "a@example.com")
            results.append("ok")
        except DuplicateUser:
            results.append("duplicate")
        except Exception as exc:
            results.append(type(exc).__name__)
    ts = [threading.Thread(target=worker) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("ok") == 1
    assert set(results) <= {"ok", "duplicate"}, results
'''

RACE_EXPERIMENT = '''import threading
from registry.service import UserRegistry, DuplicateUser
reg = UserRegistry(); b = threading.Barrier(8); out = []
def w():
    b.wait()
    try: reg.register("bob", "b@x.com"); out.append("ok")
    except DuplicateUser: out.append("dup")
    except Exception as e: out.append(type(e).__name__)
ts = [threading.Thread(target=w) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
print(sorted(out))
'''

IDS_EXPERIMENT = '''from registry.service import UserRegistry
r = UserRegistry()
print([r.register(f"u{i}", "x@x.com")["id"] for i in range(5)])
'''

LOCK_PATCH = {"edits": [
    {"path": "registry/service.py", "old": "import time", "new": "import threading\nimport time"},
    {"path": "registry/service.py", "old": "        self._next_id = 1\n\n    def _insert",
     "new": "        self._next_id = 1\n        self._lock = threading.Lock()\n\n    def _insert"},
    {"path": "registry/service.py",
     "old": "        if username in self._rows:\n            raise DuplicateUser(username)\n        time.sleep(0.001)  # simulated round-trip to the database\n        return self._insert(username, email)",
     "new": "        with self._lock:\n            if username in self._rows:\n                raise DuplicateUser(username)\n            time.sleep(0.001)  # simulated round-trip to the database\n            return self._insert(username, email)"}],
    "explanation": "Make check-then-insert atomic with a lock."}


def registry_script():
    return [
        {"name": "concurrent_dup", "code": REPRO},
        {"tool": "read_file", "args": {"path": "registry/service.py"}},
        {"done": True, "findings": "registry/service.py register() checks for the username, sleeps, then inserts: not atomic."},
        {"hypotheses": [
            {"claim": "register() is a non-atomic check-then-insert, so concurrent duplicates hit IntegrityError",
             "prediction": "with 8 simultaneous registrations of 'bob' several IntegrityError appear",
             "experiment": {"kind": "python", "code": RACE_EXPERIMENT}},
            {"claim": "ids are generated non-uniquely for different users",
             "prediction": "ids repeat for sequential distinct users",
             "experiment": {"kind": "python", "code": IDS_EXPERIMENT}}]},
        {"verdict": "supported", "reason": "output lists IntegrityError entries"},
        {"verdict": "refuted", "reason": "ids are 1..5 and unique"},
        LOCK_PATCH,
        {"approve": True, "concerns": [], "root_cause": "register() was not atomic; added a lock around check+insert."},
    ]


def make(case, tmp_path, llm, **kw):
    cfg = Config(repo=CASES / case / "repo", bug_report=(CASES / case / "BUG_REPORT.md").read_text(),
                 out_dir=tmp_path / "out", repro_runs=3, stress_runs=5, test_timeout=30, verbose=False, **kw)
    return Debugger(cfg, llm)


def test_registry_full_pipeline(tmp_path):
    llm = ScriptedLLM(registry_script())
    dbg = make("user_registry", tmp_path, llm)
    try:
        out = dbg.run()
        assert out.status is State.DONE, out.reason
        assert out.confidence == "high"
        assert "threading.Lock" in out.diff and "tests/test_agent_concurrent_dup.py" in out.diff
        assert dbg.repro["failed_runs"] >= 1                       # bug really reproduced first
        assert [h.status for h in dbg.ledger] == ["supported", "refuted"]
        assert holdout_passes(CASES / "user_registry", dbg) is True   # hidden tests pass on the patched code
        # original repo untouched
        assert "Lock" not in (CASES / "user_registry/repo/registry/service.py").read_text()
    finally:
        dbg.sb.cleanup()
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Hypothesis ledger" in report and "Prediction (logged before running)" in report
    assert "IntegrityError" in report                               # verbatim evidence from tool output
    events = Trace.load(tmp_path / "out" / "trace.jsonl")
    kinds = [e["kind"] for e in events]
    assert kinds.index("experiment_planned") < kinds.index("hypothesis_verdict")   # prediction logged first
    states = [e["data"]["state"] for e in events if e["kind"] == "state"]
    assert states == ["REPRODUCING", "INVESTIGATING", "HYPOTHESIZING", "EXPERIMENTING", "PATCHING",
                      "VERIFYING", "REVIEWING", "DONE"]


PAG_FIX = {"edits": [{"path": "paginator.py", "old": "    start = page * size", "new": "    start = (page - 1) * size"}],
           "explanation": "pages are 1-indexed"}
PAG_HYP = {"hypotheses": [{
    "claim": "paginate() treats the 1-indexed page as 0-indexed when computing the slice start",
    "prediction": "paginate([1,2,3,4,5],1,2) prints [3, 4]",
    "experiment": {"kind": "python", "code": "from paginator import paginate\nprint(paginate([1,2,3,4,5],1,2))"}}]}


def test_pagination_existing_failing_test_and_cheating_blocked(tmp_path):
    cheat = {"edits": [{"path": "tests/test_paginator.py", "old": "[1, 2]", "new": "[3, 4]"}]}   # tries to edit the test!
    llm = ScriptedLLM([
        {"done": True, "findings": "paginator.py paginate start index"},
        PAG_HYP, {"verdict": "supported", "reason": "prints [3, 4]"},
        cheat, PAG_FIX,
        {"approve": True, "concerns": [], "root_cause": "off-by-one: 1-indexed page used as 0-indexed"}])
    dbg = make("pagination", tmp_path, llm)
    try:
        out = dbg.run()
        assert out.status is State.DONE, out.reason
        assert dbg.sb.integrity_violations() == []
        assert holdout_passes(CASES / "pagination", dbg) is True
    finally:
        dbg.sb.cleanup()
    assert any("protected" in (e["data"].get("output") or "") for e in dbg.trace.records if e["kind"] == "tool_call")


def test_wrong_fix_fails_verification_then_second_iteration_succeeds(tmp_path):
    bad = {"edits": [{"path": "paginator.py", "old": "    start = page * size", "new": "    start = (page + 1) * size"}]}
    llm = ScriptedLLM([
        {"done": True, "findings": "paginator.py"},
        PAG_HYP, {"verdict": "supported", "reason": "ok"}, bad,
        PAG_HYP, {"verdict": "supported", "reason": "ok"}, PAG_FIX,
        {"approve": True, "concerns": [], "root_cause": "x"}])
    dbg = make("pagination", tmp_path, llm)
    try:
        out = dbg.run()
        assert out.status is State.DONE and out.stats["iterations"] == 2
        assert len(dbg.memory) == 1 and "failed verification" in dbg.memory[0]["summary"]
        assert "(page + 1)" not in out.diff                         # bad patch was reverted
    finally:
        dbg.sb.cleanup()


def test_baseline_mode_skips_hypotheses_and_reports_low_confidence(tmp_path):
    llm = ScriptedLLM([{"done": True, "findings": "paginator.py start index is wrong"}, PAG_FIX,
                       {"approve": True, "concerns": [], "root_cause": "off by one"}])
    dbg = make("pagination", tmp_path, llm, baseline=True)
    try:
        out = dbg.run()
        assert out.status is State.DONE and out.confidence.startswith("low")
        assert out.stats["llm_calls"] == 3
    finally:
        dbg.sb.cleanup()


def test_abstains_when_bug_cannot_be_reproduced(tmp_path):
    useless = {"name": "passes", "code": "def test_nothing():\n    assert True\n"}
    dbg = make("user_registry", tmp_path, ScriptedLLM([useless, {**useless, "name": "p2"}, {**useless, "name": "p3"}]))
    try:
        out = dbg.run()
        assert out.status is State.ABSTAINED and out.diff == ""
    finally:
        dbg.sb.cleanup()
    assert "could not be reproduced" in (tmp_path / "out" / "report.md").read_text()


def test_budget_exhaustion_is_reported_not_crashed(tmp_path):
    dbg = make("user_registry", tmp_path, ScriptedLLM(registry_script()), max_llm_calls=3)
    try:
        out = dbg.run()
        assert out.status is State.BUDGET_EXHAUSTED and out.diff == ""
    finally:
        dbg.sb.cleanup()
