"""Command line interface:  python -m autodebug REPO --bug "..."  """
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .agent import Config, Debugger
from .llm import LocalLLM
from .state import State


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autodebug", description="Autonomous debugger on a local LLM (no Docker).")
    p.add_argument("repo", type=Path, help="path to the buggy repository (it is copied, never modified in place)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bug", help="bug report text")
    g.add_argument("--bug-file", type=Path, help="file containing the bug report")
    p.add_argument("--selector", help="known failing test, e.g. tests/test_x.py::test_y (skips repro-test writing)")
    p.add_argument("--model", default=os.environ.get("AUTODEBUG_MODEL", "qwen2.5-coder:7b"))
    p.add_argument("--base-url", default=os.environ.get("AUTODEBUG_BASE_URL", "http://localhost:11434/v1"),
                   help="OpenAI-compatible endpoint (Ollama :11434, llama.cpp :8080, LM Studio :1234)")
    p.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if your server supports it)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--sandbox", choices=["auto", "bwrap", "unshare", "plain"], default="auto")
    p.add_argument("--python", help="python interpreter used to run the target's tests (default: this one)")
    p.add_argument("--baseline", action="store_true", help="ablation: skip the hypothesis/experiment stage")
    p.add_argument("--strict-review", action="store_true", help="reviewer rejection blocks the patch")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--max-llm-calls", type=int, default=60)
    p.add_argument("--max-seconds", type=float, default=1800)
    p.add_argument("--repro-runs", type=int, default=5)
    p.add_argument("--stress-runs", type=int, default=10)
    p.add_argument("--ctx-chars", type=int, default=24000, help="prompt budget in characters (~4 chars/token)")
    p.add_argument("--out", type=Path, default=Path("autodebug-runs"), help="output directory root")
    p.add_argument("--apply", action="store_true", help="copy the verified fix into REPO when done")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    bug = a.bug if a.bug is not None else a.bug_file.read_text(encoding="utf-8")
    out_dir = (a.out / time.strftime("%Y%m%d-%H%M%S")).resolve()
    cfg = Config(repo=a.repo, bug_report=bug, out_dir=out_dir, selector=a.selector, baseline=a.baseline,
                 strict_review=a.strict_review, max_iterations=a.max_iterations, max_llm_calls=a.max_llm_calls,
                 max_seconds=a.max_seconds, repro_runs=a.repro_runs, stress_runs=a.stress_runs,
                 ctx_chars=a.ctx_chars, sandbox_backend=a.sandbox, python=a.python, verbose=not a.quiet)
    llm = LocalLLM(a.model, a.base_url, temperature=a.temperature, json_mode=a.json_mode)
    dbg = Debugger(cfg, llm)
    try:
        outcome = dbg.run()
        if outcome.status is State.DONE and a.apply:
            changed = dbg.sb.export_changes(a.repo)
            print(f"applied to {a.repo}: {', '.join(changed)}", file=sys.stderr)
    finally:
        dbg.sb.cleanup()
    print(f"\nstatus: {outcome.status.value}  ({outcome.reason})\nconfidence: {outcome.confidence}\n"
          f"report: {outcome.report_path}\npatch:  {outcome.patch_path}\ntrace:  {outcome.trace_path}")
    return 0 if outcome.status is State.DONE else 1


if __name__ == "__main__":
    raise SystemExit(main())
