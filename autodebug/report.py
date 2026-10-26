"""Markdown report. Evidence blocks are copied verbatim from tool output recorded in the trace;
the LLM's prose is only ever used for claims, never as the evidence itself."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .sandbox import clip

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Debugger, Outcome


def _fence(text: str, lang: str = "") -> str:
    cleaned = text.rstrip()
    ticks = 3
    for m in re.finditer(r"`+", cleaned):
        ticks = max(ticks, len(m.group(0)) + 1)
    f = "`" * ticks
    return f"{f}{lang}\n{cleaned}\n{f}"


def render_report(dbg: "Debugger", out: "Outcome") -> str:
    cfg, s = dbg.cfg, out.stats
    L: list[str] = [f"# AutoDebug report: {out.status.value}", ""]
    L += [f"- **Result:** {out.reason}",
          f"- **Confidence:** {out.confidence}",
          f"- **Model:** `{s['model']}` | **sandbox:** `{s['sandbox']}` | **mode:** {'baseline (no hypothesis stage)' if cfg.baseline else 'hypothesis-driven'}",
          f"- **Cost:** {s['llm_calls']} LLM calls, {s['seconds']}s, {s['iterations']} iteration(s), {s['hypotheses']} hypothesis(es)",
          f"- **Full trace:** `{out.trace_path}` (every claim below cites an event id `#n`)", ""]
    
    # Blockquote the bug report so markdown headings inside it don't inject fake sections (BUG-083)
    quoted_bug = "\n".join(f"> {line}" if line.strip() else ">" for line in cfg.bug_report.strip().splitlines())
    L += ["## Bug report", "", quoted_bug, ""]

    L += ["## Reproduction", ""]
    if dbg.repro:
        r = dbg.repro
        L += [f"`{r['selector'] or 'full test suite'}` failed in **{r['failed_runs']}/{r['runs']}** runs (evidence #{r['evidence']}).", ""]
        ev = dbg.trace.get(r["evidence"])
        if ev:
            L += [_fence(clip(ev["data"].get("output", ""), 1800)), ""]
    else:
        L += ["The bug could not be reproduced with a failing test, so no fix was attempted.", ""]

    if out.root_cause:
        L += ["## Root cause", "", out.root_cause, ""]

    L += ["## Hypothesis ledger", ""]
    if dbg.ledger:
        L += ["| id | iter | status | claim | evidence |", "|---|---|---|---|---|"]
        for h in dbg.ledger:
            ev_str = ", ".join(f"#{e}" for e in h.evidence_events) or "-"
            # Replace newlines with space and escape pipes (BUG-082, BUG-094)
            clean_claim = h.claim.replace("\r", " ").replace("\n", " ").replace("|", r"\|")[:140]
            L.append(f"| {h.id} | {h.iteration} | **{h.status}** | {clean_claim} | {ev_str} |")
        L.append("")
        for h in dbg.ledger:
            if not h.evidence_events:
                continue
            L += [f"### {h.id}: {h.status}", "", f"**Prediction (logged before running):** {h.prediction}", ""]
            for e in h.evidence_events:
                rec = dbg.trace.get(e)
                if rec:
                    L += [f"Evidence #{e} ({rec['data'].get('tool')}):", _fence(clip(rec["data"].get("output", ""), 1500)), ""]
            L += [f"**Verdict reason:** {h.reason}", ""]
    else:
        L += ["No hypotheses were generated.", ""]

    if out.diff:
        L += ["## Patch (includes the regression test)", "", _fence(out.diff, "diff"), ""]
        if dbg.patch_explanation:
            L += [f"**Explanation:** {dbg.patch_explanation}", ""]
        v = dbg.verification
        L += ["## Verification", "",
              f"- Repro test passed **{v.get('repro_runs_passed')}** runs after the fix (evidence #{v.get('repro_evidence')})",
              f"- Full suite: no new failures: **{not v.get('new_failures')}** (evidence #{v.get('full_suite_evidence')})",
              f"- Existing tests/config untouched: **{v.get('protected_files_intact')}**", ""]
        if dbg.review:
            L += ["## Reviewer", "", f"Approved: **{dbg.review['approve']}**"]
            L += [f"- concern: {c}" for c in dbg.review["concerns"]] + [""]
    else:
        L += ["## Patch", "", "No verified patch was produced.", ""]

    if dbg.memory:
        L += ["## Failed attempts (kept as memory for later iterations)", ""]
        L += [f"- {m['summary']}" for m in dbg.memory] + [""]
    L += ["## Caveats", "",
          "- A passing test suite is evidence, not proof. Review the patch before merging.",
          f"- Sandbox backend `{s['sandbox']}`: " + ("strong isolation." if s["sandbox"] == "bwrap" else
           "no filesystem isolation from your user account; do not run untrusted repositories." if s["sandbox"] == "plain" else
           "network isolation only; the process can still read files your user can read."), ""]
    return "\n".join(L)
