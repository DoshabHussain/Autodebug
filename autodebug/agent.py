"""The debugging agent.

Roles (Planner / Investigator / Experimenter / Patcher / Tester / Reviewer) are *states of one
orchestrator*, each with its own prompt and context slice, not separate autonomous LLM agents.
The orchestrator, not the model, decides what happens next, so a small local model cannot wander.

    REPRODUCING -> INVESTIGATING -> HYPOTHESIZING -> EXPERIMENTING -> PATCHING -> VERIFYING -> REVIEWING -> DONE
                                         ^________________ (refuted / failed fix: loop with memory) ____|
    any state -> ABSTAINED (honest give-up) | BUDGET_EXHAUSTED
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import BaseLLM, LLMError
from .sandbox import Sandbox, SandboxError, clip, is_protected
from .state import TERMINAL, Budget, BudgetExceeded, Hypothesis, State, StateMachine, Trace
from .tools import READONLY_TOOLS, Tools

SYSTEM = """You are an expert Python debugger working through a restricted tool layer.
Reply with exactly ONE JSON object and nothing else: no markdown fences, no commentary.
Never claim tool output you have not been shown. Existing tests are read-only: fix the source code, never the tests."""

TOOL_HELP = """Tools (reply {"tool": NAME, "args": {...}, "thought": "why"}):
- list_files {"path": "."}
- read_file {"path": "pkg/mod.py", "start": 1, "end": 120}
- search_code {"text": "literal text", "path": "."}
- run_tests {"selector": "tests/test_x.py::test_y", "runs": 1}   (selector optional; pytest)
- run_python {"code": "print(1)"}   (throw-away experiment script run from the repo root; edits are discarded)
- show_diff {}
When you know enough, reply {"done": true, "findings": "<suspect files/functions and what you observed>"}."""


@dataclass
class Config:
    repo: Path
    bug_report: str
    out_dir: Path
    selector: str | None = None          # optional known-failing test to use as the reproduction
    baseline: bool = False               # ablation: skip the hypothesis/experiment stage
    strict_review: bool = False
    max_iterations: int = 3
    max_llm_calls: int = 60
    max_seconds: float = 1800
    investigate_steps: int = 8
    repro_attempts: int = 3
    repro_runs: int = 5                  # runs used to measure how often the bug reproduces
    stress_runs: int = 10                # runs of the repro test required to pass after the fix
    ctx_chars: int = 24000
    max_changed_lines: int = 80
    sandbox_backend: str = "auto"
    python: str | None = None
    test_timeout: float = 120
    verbose: bool = True


@dataclass
class Outcome:
    status: State
    reason: str
    diff: str = ""
    root_cause: str = ""
    confidence: str = "n/a"
    report_path: Path | None = None
    patch_path: Path | None = None
    trace_path: Path | None = None
    stats: dict[str, Any] = field(default_factory=dict)


class Debugger:
    def __init__(self, cfg: Config, llm: BaseLLM, sandbox: Sandbox | None = None) -> None:
        self.cfg, self.llm = cfg, llm
        # BUG-058: Clip bug report to a fraction of context budget
        self.cfg.bug_report = clip(self.cfg.bug_report, max(1000, self.cfg.ctx_chars // 3))
        # BUG-065: Initialize Sandbox BEFORE creating out_dir and Trace
        self.sb = sandbox or Sandbox(cfg.repo, backend=cfg.sandbox_backend, python=cfg.python,
                                     timeout=cfg.test_timeout, exclude=[cfg.out_dir])
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(cfg.out_dir / "trace.jsonl")
        self.tools = Tools(self.sb, self.trace, test_timeout=cfg.test_timeout,
                           max_changed_lines=cfg.max_changed_lines)
        self.sm = StateMachine()
        self.budget = Budget(cfg.max_llm_calls, cfg.max_seconds, cfg.max_iterations)
        self.ledger: list[Hypothesis] = []
        self.memory: list[dict[str, str]] = []      # failed patch attempts, fed back to the model
        self.repro: dict[str, Any] = {}
        self.repro_error = ""
        self.baseline_failed: set[str] = set()
        self.failure_output = ""
        self.findings = ""
        self.verification: dict[str, Any] = {}
        self.review: dict[str, Any] = {}
        self.chosen: Hypothesis | None = None
        self.iteration = 0
        self.patch_explanation = ""
        self._hyp_feedback = ""

    # ------------------------------------------------------------------ plumbing
    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[{self.sm.state.value:<13}] {msg}", file=sys.stderr, flush=True)

    def _go(self, state: State) -> None:
        self.sm.to(state)
        self.trace.state = state.value
        self.trace.emit("state", state=state.value, iteration=self.iteration)

    def _sync_tool_budget(self) -> None:
        """Enforce max-seconds by capping tool execution to remaining time budget (BUG-057)."""
        rem = self.budget.max_seconds - self.budget.elapsed
        self.tools.test_timeout = max(1.0, min(self.cfg.test_timeout, rem))

    def _ask(self, messages: list[dict[str, str]], purpose: str) -> dict[str, Any]:
        self.budget.check()
        before = self.llm.calls
        remaining = self.budget.max_llm_calls - self.budget.llm_calls
        retries = max(0, min(2, remaining - 1))
        try:
            data = self.llm.chat_json(self._trim(messages), retries=retries)
        finally:
            self.budget.charge(self.llm.calls - before)
        self.trace.emit("llm_call", purpose=purpose, prompt_tail=clip(messages[-1]["content"], 1500),
                        response=clip(json.dumps(data), 3000))
        return data

    def _trim(self, msgs: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep system+first user message, then recent messages in pairs to avoid consecutive user roles (BUG-061)."""
        if len(msgs) <= 2:
            return msgs
        head, rest = msgs[:2], msgs[2:]
        budget = self.cfg.ctx_chars - sum(len(m["content"]) for m in head)
        while rest and sum(len(m["content"]) for m in rest) > budget:
            if len(rest) >= 2:
                rest = rest[2:]
            else:
                rest = []
        return head + rest

    def _digest(self) -> str:
        """File list plus class/def signatures: a cheap map of the codebase for small context windows."""
        files = self.sb.files()
        lines = ["FILES:", *files[:80]]
        sig = re.compile(r"^\s*(class|def|async def)\s+\w+")
        for rel in files:
            if rel.endswith(".py") and not is_protected(rel):
                try:
                    text = (self.sb.ws / rel).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                sigs = [l.rstrip() for l in text.splitlines() if sig.match(l)][:25]
                if sigs:
                    lines += [f"\n# {rel}", *sigs]
        limit = max(100, self.cfg.ctx_chars // 4)
        return clip("\n".join(lines), limit)

    # ------------------------------------------------------------------ main loop
    def run(self) -> Outcome:
        self.trace.emit("start", bug_report=self.cfg.bug_report, model=getattr(self.llm, "model", "?"),
                        sandbox=self.sb.backend, baseline=self.cfg.baseline)
        self._log(f"sandbox backend: {self.sb.backend}")
        try:
            self._go(State.REPRODUCING)
            if not self._reproduce():
                return self._finish(State.ABSTAINED, self.repro_error or "could not reproduce the bug with a failing test")
            self._go(State.INVESTIGATING)
            self._investigate()
            while True:
                if self.iteration >= self.cfg.max_iterations:
                    return self._finish(State.ABSTAINED, "iteration budget used up without a verified fix")
                self.iteration += 1
                if self.cfg.baseline:
                    self.chosen = Hypothesis(f"B{self.iteration}", self.findings or "(no findings)", "n/a", {}, "inconclusive",
                                             "baseline mode: no hypothesis testing", iteration=self.iteration)
                    self.ledger.append(self.chosen)
                    if self.sm.state is not State.INVESTIGATING:
                        self._go(State.HYPOTHESIZING)
                else:
                    self._go(State.HYPOTHESIZING)
                    hyps = self._hypothesize()
                    if not hyps:
                        continue
                    self._go(State.EXPERIMENTING)
                    self.chosen = self._experiment(hyps)
                    if self.chosen is None:
                        continue
                self._go(State.PATCHING)
                if not self._patch(self.chosen):
                    continue
                self._go(State.VERIFYING)
                if not self._verify(self.chosen):
                    continue
                self._go(State.REVIEWING)
                is_approved = self._review(self.chosen)
                if is_approved or not self.cfg.strict_review:
                    reason = "verified fix produced" if is_approved else "verified fix produced (rejected by reviewer)"
                    return self._finish(State.DONE, reason)
                self._remember("reviewer rejected the patch: " + "; ".join(self.review.get("concerns", [])))
                self.sb.revert_changes()
        except BudgetExceeded as e:
            return self._finish(State.BUDGET_EXHAUSTED, str(e))
        except LLMError as e:
            return self._finish(State.ABSTAINED, f"LLM failure: {e}")
        except SandboxError as e:
            return self._finish(State.ABSTAINED, f"sandbox failure: {e}")
        except KeyboardInterrupt:
            return self._finish(State.ABSTAINED, "interrupted by user")
        except Exception as e:
            return self._finish(State.ABSTAINED, f"unexpected error: {e}")
        finally:
            self.trace.close()

    # ------------------------------------------------------------------ REPRODUCING
    def _reproduce(self) -> bool:
        t = self.tools
        self._sync_tool_budget()
        sel = self.cfg.selector
        if sel:
            rate = t.call("run_tests", {"selector": sel, "runs": self.cfg.repro_runs})
            codes = rate.data.get("codes", [])
            if any(c in (4, 5) for c in codes):
                self.repro_error = f"selector error (pytest exit code {codes[0]}): {rate.output}"
                self._log(self.repro_error)
                return False
            if not rate.ok or rate.data.get("failed_runs", 0) == 0:
                self.repro_error = f"selector does not fail ({rate.data.get('failed_runs', 0)}/{rate.data.get('runs', 1)} failed)"
                self._log(self.repro_error)
                return False
            self.failure_output = rate.data.get("excerpt", rate.output)
            self.repro = {"selector": sel, "runs": rate.data["runs"], "failed_runs": rate.data["failed_runs"],
                          "evidence": rate.event_id}
            self._log(f"reproduced with given selector: {sel}")
            full = t.call("run_tests", {"runs": 1})
            self.baseline_failed = set(full.data.get("failed_ids", []))
            return True

        base = t.call("run_tests", {"runs": 1})
        if not base.ok:
            self._log(base.output)
            return False

        if base.data.get("failed_runs", 0) > 0:
            ids = base.data.get("failed_ids", [])
            sel = ids[0] if ids else None
            self.failure_output = base.data["excerpt"]
            self.trace.emit("reproduction", mode="existing failing test", selector=sel, evidence=base.event_id)
            self._log(f"suite already fails: {sel or 'full suite'}")
        elif base.data.get("error_runs", 0) > 0:
            # Collection or setup error (BUG-046)
            self.failure_output = base.data["excerpt"]
            self._log(f"suite errors during collection: {base.data['error_runs']} errored")
            sel = self._write_repro_test(suite_errored=True, error_output=base.data["excerpt"])
            if sel is None:
                return False
        else:
            sel = self._write_repro_test(suite_errored=False)
            if sel is None:
                return False

        rate = t.call("run_tests", {"selector": sel, "runs": self.cfg.repro_runs}) if sel else \
            t.call("run_tests", {"runs": self.cfg.repro_runs})
        if not rate.ok or rate.data["failed_runs"] == 0:
            self._log("the reproduction does not fail; abstaining")
            return False
        self.failure_output = rate.data["excerpt"]
        self.repro = {"selector": sel, "runs": rate.data["runs"], "failed_runs": rate.data["failed_runs"],
                      "evidence": rate.event_id}
        self._log(f"reproduced: {sel or 'full suite'} fails {rate.data['failed_runs']}/{rate.data['runs']} runs")
        full = t.call("run_tests", {"runs": 1})            # baseline *with* the repro test present
        self.baseline_failed = set(full.data.get("failed_ids", []))
        return True

    def _write_repro_test(self, suite_errored: bool = False, error_output: str = "") -> str | None:
        if suite_errored:
            status_text = f"The existing test suite ERRORS during collection or run:\n{error_output}\n"
        else:
            status_text = "The existing test suite PASSES, so the bug is not covered.\n"

        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"BUG REPORT:\n{self.cfg.bug_report}\n\n{status_text}"
                    f"CODEBASE MAP:\n{self._digest()}\n\n"
                    "Write ONE pytest test that FAILS on the current buggy code and would pass once fixed. "
                    "If the bug is timing/concurrency related, use threads with a threading.Barrier so the race is "
                    "triggered reliably. Assert on correct behaviour; do not assert on the bug itself.\n"
                    'Reply {"name": "short_snake_case", "code": "<python source of the test file>"}.'}]
        for attempt in range(self.cfg.repro_attempts):
            d = self._ask(msgs, f"write_repro_test#{attempt + 1}")
            msgs.append({"role": "assistant", "content": json.dumps(d)})
            res = self.tools.call("create_test", {"name": d.get("name", f"repro{attempt}"),
                                                  "code": d.get("code", "")}) if isinstance(d.get("name", ""), str) \
                else None
            if res is None or not res.ok:
                msgs.append({"role": "user", "content": f"Could not create the test: {res.output if res else 'bad name'}. Try again."})
                continue
            rel = res.data["path"]
            self._sync_tool_budget()
            run = self.tools.call("run_tests", {"selector": rel, "runs": self.cfg.repro_runs})
            out = run.data.get("excerpt", run.output)
            # Filter out syntax errors, missing modules, and test-local NameError/AttributeError (BUG-045)
            infra = (
                re.search(r"(SyntaxError|ModuleNotFoundError|ImportError|NameError|AttributeError|FixtureLookupError)", out)
                and not re.search(r"(SyntaxError|ModuleNotFoundError|ImportError|NameError|AttributeError)", self.cfg.bug_report)
            )
            if run.ok and run.data["failed_runs"] >= 1 and run.data["error_runs"] == 0 and not infra:
                self.trace.emit("reproduction", mode="agent-written test", selector=rel, evidence=run.event_id)
                self.failure_output = out
                return rel
            why = ("it PASSED every run, but it must fail on the buggy code" if run.ok and run.data["failed_runs"] == 0
                   else "it failed for an infrastructure reason or test typo (NameError/Import/Syntax), not the bug")
            msgs.append({"role": "user", "content": f"Your test {why}.\nOutput:\n{clip(out, 2500)}\n"
                                                    "Write a different test with a NEW unique name."})
            self.tools.sb.resolve(rel).unlink(missing_ok=True)    # discard invalid repro attempts
        return None

    # ------------------------------------------------------------------ INVESTIGATING
    def _investigate(self) -> None:
        msgs = [{"role": "system", "content": SYSTEM + "\n\n" + TOOL_HELP},
                {"role": "user", "content":
                    f"BUG REPORT:\n{self.cfg.bug_report}\n\nREPRODUCTION: {self.repro.get('selector') or 'full suite'} "
                    f"fails {self.repro.get('failed_runs')}/{self.repro.get('runs')} runs.\n"
                    f"FAILURE OUTPUT:\n{clip(self.failure_output, 3000)}\n\nCODEBASE MAP:\n{self._digest()}\n\n"
                    "Investigate with the tools. Read the suspicious code. Be efficient."}]
        for step in range(self.cfg.investigate_steps):
            d = self._ask(msgs, f"investigate#{step + 1}")
            if d.get("done"):
                self.findings = str(d.get("findings", ""))[:3000]
                break
            name, args = d.get("tool"), d.get("args") or {}
            msgs.append({"role": "assistant", "content": json.dumps(d)})
            if name not in READONLY_TOOLS:
                msgs.append({"role": "user", "content": f"Tool {name!r} is not available here. Use one of: {READONLY_TOOLS}."})
                continue
            self._sync_tool_budget()
            res = self.tools.call(name, args)
            self._log(f"{name} {clip(str(args), 80)} -> ok={res.ok}")
            msgs.append({"role": "user", "content": f"TOOL RESULT ({name}, ok={res.ok}):\n{clip(res.output, 3000)}"})
        else:
            msgs.append({"role": "user", "content": 'Step limit reached. Reply now with {"done": true, "findings": "..."}.'})
            self.findings = str(self._ask(msgs, "investigate#final").get("findings", ""))[:3000]
        self.trace.emit("findings", text=self.findings)

    # ------------------------------------------------------------------ HYPOTHESIZING
    def _hypothesize(self) -> list[Hypothesis]:
        refuted = [f"- {h.id}: {h.claim} (refuted: {h.reason})" for h in self.ledger if h.status == "refuted"]
        failed = [f"- attempt {i + 1}: {m['summary']}\n  diff:\n{clip(m.get('diff', ''), 800)}"
                  for i, m in enumerate(self.memory)]
        feedback_section = f"\nFEEDBACK FROM PREVIOUS ATTEMPT:\n{self._hyp_feedback}\n" if self._hyp_feedback else ""
        prompt = (f"BUG REPORT:\n{self.cfg.bug_report}\n\nFAILURE OUTPUT:\n{clip(self.failure_output, 2000)}\n\n"
                  f"FINDINGS SO FAR:\n{self.findings}\n\n"
                  f"ALREADY REFUTED:\n{chr(10).join(refuted) or '(none)'}\n\n"
                  f"FAILED FIX ATTEMPTS:\n{chr(10).join(failed) or '(none)'}\n"
                  f"{feedback_section}\n"
                  "Propose 2-4 COMPETING, falsifiable hypotheses about the root cause. For each give:\n"
                  '  "claim": what is wrong;\n'
                  '  "prediction": the concrete output the experiment will show IF the claim is true (and what it would show if false);\n'
                  '  "experiment": {"kind": "python", "code": "<self-contained script, run from the repo root, that PRINTS observations>"} '
                  'or {"kind": "tests", "selector": "tests/test_x.py::test_y", "runs": 5}.\n'
                  "Experiments should isolate ONE factor (e.g. vary concurrency, inspect state) and must not edit files.\n"
                  'Reply {"hypotheses": [ ... ]}.')
        d = self._ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], "hypothesize")
        out: list[Hypothesis] = []
        raw_hyps = d.get("hypotheses")
        # Handle dict returned instead of list (BUG-059)
        if isinstance(raw_hyps, dict):
            raw_hyps = [raw_hyps]
        elif not isinstance(raw_hyps, list):
            raw_hyps = []

        for raw in raw_hyps[:4]:
            exp = raw.get("experiment") if isinstance(raw, dict) else None
            if not (isinstance(raw, dict) and isinstance(raw.get("claim"), str) and isinstance(exp, dict)
                    and exp.get("kind") in ("python", "tests")):
                continue
            h = Hypothesis(f"H{len(self.ledger) + 1}", raw["claim"][:500], str(raw.get("prediction", ""))[:500],
                           exp, iteration=self.iteration)
            self.ledger.append(h)
            out.append(h)
            self.trace.emit("hypothesis", id=h.id, claim=h.claim, prediction=h.prediction, experiment=clip(str(exp), 1500))
        
        if not out:
            # Provide feedback if invalid hypotheses were returned (BUG-060)
            self._hyp_feedback = "Your previous hypotheses were invalid or missing 'claim' / 'experiment' objects. Provide valid hypotheses."
        else:
            self._hyp_feedback = ""
        self._log(f"{len(out)} hypotheses: " + "; ".join(f"{h.id}={clip(h.claim, 60)}" for h in out))
        return out

    # ------------------------------------------------------------------ EXPERIMENTING
    def _experiment(self, hyps: list[Hypothesis]) -> Hypothesis | None:
        for h in hyps:
            self.trace.emit("experiment_planned", id=h.id, prediction=h.prediction)
            self._sync_tool_budget()
            if h.experiment["kind"] == "python":
                res = self.tools.call("run_python", {"code": h.experiment.get("code", "")})
            else:
                res = self.tools.call("run_tests", {"selector": h.experiment.get("selector"),
                                                    "runs": h.experiment.get("runs", 3)})
            h.evidence_events.append(res.event_id)
            verdict = self._ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                f"HYPOTHESIS: {h.claim}\nPREDICTION (written before running): {h.prediction}\n\n"
                f"EXPERIMENT OUTPUT (verbatim, ok={res.ok}):\n{clip(res.output, 3000)}\n\n"
                "Judge STRICTLY from the output above. 'supported' only if it matches the prediction, "
                "'refuted' if it contradicts it, otherwise 'inconclusive'.\n"
                'Reply {"verdict": "supported|refuted|inconclusive", "reason": "..."}.'}], f"judge {h.id}")
            v = str(verdict.get("verdict", "")).lower()
            h.status = v if v in ("supported", "refuted", "inconclusive") else "inconclusive"
            h.reason = str(verdict.get("reason", ""))[:400]
            self.trace.emit("hypothesis_verdict", id=h.id, status=h.status, reason=h.reason, evidence=res.event_id)
            self._log(f"{h.id}: {h.status} ({clip(h.reason, 80)})")
        for wanted in ("supported", "inconclusive"):
            for h in hyps:
                if h.status == wanted:
                    return h
        return None

    # ------------------------------------------------------------------ PATCHING
    def _context_files(self) -> list[str]:
        text = self.failure_output + "\n" + self.findings + "\n" + (self.chosen.claim if self.chosen else "")
        cands = re.findall(r"([\w./\-]+\.py)", text) + self.tools.files_read
        seen: list[str] = []
        for item in cands:
            c = item.removeprefix("./")
            try:
                ok = self.sb.resolve(c).is_file() and not is_protected(c)
            except SandboxError:
                ok = False
            if ok and c not in seen:
                seen.append(c)
        return seen[:3]

    def _patch(self, h: Hypothesis) -> bool:
        files = self._context_files()
        per_file = max(2000, self.cfg.ctx_chars // 6)
        # BUG-062: Read with errors='replace' to tolerate Latin-1 / non-utf-8 files
        blobs = "\n\n".join(f"=== {f} ===\n{clip((self.sb.ws / f).read_text(encoding='utf-8', errors='replace'), per_file)}" for f in files)
        evidence = "\n".join(clip((self.trace.get(e) or {}).get("data", {}).get("output", ""), 1500)
                             for e in h.evidence_events)
        failed = "\n".join(f"- {m['summary']}" for m in self.memory) or "(none)"
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"BUG REPORT:\n{self.cfg.bug_report}\n\nROOT CAUSE HYPOTHESIS ({h.status}): {h.claim}\n"
                    f"EVIDENCE:\n{evidence or h.reason}\n\nFAILURE OUTPUT:\n{clip(self.failure_output, 1500)}\n\n"
                    f"PREVIOUS FAILED FIXES:\n{failed}\n\nSOURCE FILES:\n{blobs or '(none found; use paths from the failure output)'}\n\n"
                    "Fix the ROOT CAUSE with the smallest correct change. Do not edit tests. Reply with search/replace edits:\n"
                    '{"edits": [{"path": "pkg/mod.py", "old": "<exact existing text, unique in the file>", "new": "<replacement>"}], '
                    '"explanation": "..."}\n`old` must be copied EXACTLY (including indentation).'}]
        for attempt in range(2):
            d = self._ask(msgs, f"patch#{attempt + 1}")
            msgs.append({"role": "assistant", "content": json.dumps(d)})
            self._sync_tool_budget()
            res = self.tools.call("apply_edits", {"edits": d.get("edits")})
            if res.ok:
                self.patch_explanation = str(d.get("explanation", ""))[:800]
                self._log(res.output)
                return True
            self._log(f"patch rejected: {res.output}")
            msgs.append({"role": "user", "content": f"Patch rejected: {res.output}\nReply with corrected edits."})
        self._remember(f"patch for {h.id} could not be applied: {res.output}")
        return False

    def _remember(self, summary: str, diff: str = "") -> None:
        self.memory.append({"summary": summary, "diff": diff or self.sb.diff()})
        self.trace.emit("memory", summary=summary)

    # ------------------------------------------------------------------ VERIFYING
    def _verify(self, h: Hypothesis) -> bool:
        t, problems = self.tools, []
        violations = self.sb.integrity_violations()
        problems += violations
        diff = self.sb.diff()
        if not diff.strip():
            problems.append("patch produced no changes to the source code")
        sel = self.repro.get("selector")
        self._sync_tool_budget()
        r1 = t.call("run_tests", {"selector": sel, "runs": self.cfg.stress_runs} if sel
                    else {"runs": self.cfg.stress_runs})
        r1_pass = r1.data.get("runs", 0) - r1.data.get("failed_runs", 0) - r1.data.get("error_runs", 0)
        if not r1.ok or r1.data.get("failed_runs", 0) or r1.data.get("error_runs", 0):
            problems.append(f"repro still fails in {self.cfg.stress_runs - r1_pass}/{self.cfg.stress_runs} runs")
        r2 = t.call("run_tests", {"runs": 1})
        new_failures = sorted(set(r2.data.get("failed_ids", [])) - self.baseline_failed)
        # Check for setup/collection errors (BUG-047)
        if not r2.ok or r2.data.get("error_runs", 0) > 0:
            problems.append("full test suite errored")
        if any(c != 0 for c in r2.data.get("codes", [])) and not new_failures and not r2.data.get("error_runs", 0):
            problems.append("full test suite returned non-zero exit code")
        if new_failures:
            problems.append("new failing tests: " + ", ".join(new_failures))
        if sel and sel in r2.data.get("failed_ids", []):
            problems.append("repro test fails in the full-suite run")
        self.verification = {
            "repro_runs_passed": f"{r1_pass}/{self.cfg.stress_runs}", "repro_evidence": r1.event_id,
            "full_suite_evidence": r2.event_id, "new_failures": new_failures,
            "protected_files_intact": not violations, "problems": problems, "ok": not problems}
        self.trace.emit("verification", **self.verification)
        self._log("verification " + ("PASSED" if not problems else "FAILED: " + "; ".join(problems)))
        if problems:
            self._remember(f"fix for {h.id} failed verification: " + "; ".join(problems))
            self.sb.revert_changes()
            return False
        return True

    # ------------------------------------------------------------------ REVIEWING
    def _review(self, h: Hypothesis) -> bool:
        d = self._ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            f"BUG REPORT:\n{self.cfg.bug_report}\n\nCLAIMED ROOT CAUSE: {h.claim}\n\nPATCH:\n{clip(self.sb.diff(), 4000)}\n\n"
            f"VERIFICATION: {json.dumps(self.verification)[:800]}\n\n"
            "Review as a skeptical maintainer: does the patch address the ROOT CAUSE (not just the symptom), "
            "is it minimal, does it special-case the test, can it introduce a regression?\n"
            'Reply {"approve": true|false, "concerns": ["..."], "root_cause": "one-paragraph root cause in plain words"}.'}],
            "review")
        concerns = d.get("concerns") or []
        approve_val = d.get("approve")
        if isinstance(approve_val, str):
            approved = approve_val.strip().lower() in ("true", "1", "yes")
        elif isinstance(approve_val, bool):
            approved = approve_val
        else:
            approved = False

        self.review = {"approve": approved,
                       "concerns": [str(c)[:300] for c in concerns] if isinstance(concerns, list) else [],
                       "root_cause": str(d.get("root_cause", h.claim))[:1000]}
        self.trace.emit("review", **self.review)
        return self.review["approve"]

    # ------------------------------------------------------------------ finish
    def _confidence(self) -> str:
        h = self.chosen
        if not (h and self.verification.get("ok")):
            return "n/a"
        if self.cfg.baseline or h.status != "supported":
            return "low (root cause not confirmed by an experiment)"
        return "high" if self.review.get("approve", True) and not self.review.get("concerns") else "medium"

    def _finish(self, status: State, reason: str) -> Outcome:
        from .report import render_report
        diff = self.sb.diff() if status is State.DONE else ""
        # Require non-empty diff for DONE (BUG-049)
        if status is State.DONE and not diff.strip():
            status = State.ABSTAINED
            reason = "no changes were made to fix the bug"
            diff = ""

        if self.sm.state not in TERMINAL:
            self._go(status)
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        patch_path = out / "patch.diff"
        patch_path.write_text(diff, encoding="utf-8")
        outcome = Outcome(status, reason, diff,
                          self.review.get("root_cause") or (self.chosen.claim if self.chosen and status is State.DONE else ""),
                          self._confidence(), out / "report.md", patch_path, out / "trace.jsonl",
                          {"llm_calls": self.budget.llm_calls, "seconds": round(self.budget.elapsed, 1),
                           "iterations": self.iteration, "hypotheses": len(self.ledger),
                           "sandbox": self.sb.backend, "model": getattr(self.llm, "model", "?")})
        self.trace.emit("finish", status=status.value, reason=reason, stats=outcome.stats)
        (out / "report.md").write_text(render_report(self, outcome), encoding="utf-8")
        self.trace.close()
        self._log(f"{status.value}: {reason}")
        return outcome
