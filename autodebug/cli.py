"""Command line interface:  python -m autodebug REPO --bug "..."  """
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from . import __version__
from .agent import Config, Debugger
from .llm import LocalLLM
from .sandbox import SandboxError
from .state import State


def _positive_int(val: str) -> int:
    try:
        n = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {val!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer (> 0), got {n}")
    return n


def _positive_float(val: str) -> float:
    try:
        n = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {val!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number (> 0), got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autodebug", description="Autonomous debugger on a local LLM (no Docker).")
    p.add_argument("--version", action="version", version=f"autodebug {__version__}")
    p.add_argument("repo", type=Path, help="path to the buggy repository (it is copied, never modified in place)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bug", help="bug report text")
    g.add_argument("--bug-file", type=Path, help="file containing the bug report")
    p.add_argument("--selector", help="known failing test, e.g. tests/test_x.py::test_y (skips repro-test writing)")
    p.add_argument("--model", default=os.environ.get("AUTODEBUG_MODEL", "qwen2.5-coder:7b"))
    p.add_argument("--base-url", default=os.environ.get("AUTODEBUG_BASE_URL", "http://localhost:11434/v1"),
                   help="OpenAI-compatible endpoint (Ollama :11434, llama.cpp :8080, LM Studio :1234)")
    p.add_argument("--api-key", default=os.environ.get("AUTODEBUG_API_KEY"), help="API key for LLM endpoint")
    p.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if your server supports it)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=_positive_float, default=600.0, help="timeout per LLM call in seconds")
    p.add_argument("--max-tokens", type=_positive_int, default=2048, help="max tokens per LLM call")
    p.add_argument("--sandbox", choices=["auto", "bwrap", "unshare", "plain"], default="auto")
    p.add_argument("--python", help="python interpreter used to run the target's tests (default: this one)")
    p.add_argument("--baseline", action="store_true", help="ablation: skip the hypothesis/experiment stage")
    p.add_argument("--strict-review", action="store_true", help="reviewer rejection blocks the patch")
    p.add_argument("--max-iterations", type=_positive_int, default=3)
    p.add_argument("--max-llm-calls", type=_positive_int, default=60)
    p.add_argument("--max-seconds", type=_positive_float, default=1800.0)
    p.add_argument("--repro-runs", type=_positive_int, default=5)
    p.add_argument("--stress-runs", type=_positive_int, default=10)
    p.add_argument("--test-timeout", type=_positive_float, default=120.0, help="timeout for pytest runs in seconds")
    p.add_argument("--max-changed-lines", type=_positive_int, default=80, help="cap on total changed lines in patch")
    p.add_argument("--ctx-chars", type=_positive_int, default=24000, help="prompt budget in characters (~4 chars/token)")
    p.add_argument("--out", type=Path, default=Path("autodebug-runs"), help="output directory root")
    p.add_argument("--apply", action="store_true", help="copy the verified fix into REPO when done")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    # Validate repo path exists and is a directory (BUG-064)
    if not a.repo.exists() or not a.repo.is_dir():
        print(f"error: not a directory: {a.repo}", file=sys.stderr)
        return 2

    # Validate bug report / bug-file (BUG-066, BUG-067, BUG-069)
    if a.bug is not None:
        bug = a.bug
    else:
        if not a.bug_file.exists() or not a.bug_file.is_file():
            print(f"error: no such bug file: {a.bug_file}", file=sys.stderr)
            return 2
        try:
            bug = a.bug_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"error reading bug file: {e}", file=sys.stderr)
            return 2

    if not bug.strip():
        print("error: bug description required", file=sys.stderr)
        return 2

    # Prevent output directory collisions on runs started in same second (BUG-071)
    base_name = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (a.out / base_name).resolve()
    counter = 1
    while out_dir.exists():
        out_dir = (a.out / f"{base_name}_{counter}").resolve()
        counter += 1

    cfg = Config(repo=a.repo, bug_report=bug, out_dir=out_dir, selector=a.selector, baseline=a.baseline,
                 strict_review=a.strict_review, max_iterations=a.max_iterations, max_llm_calls=a.max_llm_calls,
                 max_seconds=a.max_seconds, repro_runs=a.repro_runs, stress_runs=a.stress_runs,
                 test_timeout=a.test_timeout, max_changed_lines=a.max_changed_lines,
                 ctx_chars=a.ctx_chars, sandbox_backend=a.sandbox, python=a.python, verbose=not a.quiet)
    llm = LocalLLM(a.model, a.base_url, temperature=a.temperature, timeout=a.timeout,
                   max_tokens=a.max_tokens, json_mode=a.json_mode, api_key=a.api_key)

    try:
        dbg = Debugger(cfg, llm)
    except SandboxError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        outcome = dbg.run()
        if outcome.status is State.DONE and a.apply:
            changed = dbg.sb.export_changes(a.repo)
            print(f"applied to {a.repo}: {', '.join(changed)}", file=sys.stderr)
    finally:
        dbg.sb.cleanup()

    print(f"\nstatus: {outcome.status.value}  ({outcome.reason})\nconfidence: {outcome.confidence}\n"
          f"report: {outcome.report_path}\npatch:  {outcome.patch_path}\ntrace:  {outcome.trace_path}")
    
    if outcome.status is State.DONE:
        # Non-zero exit code if reviewer rejected patch in non-strict mode (BUG-050)
        if dbg.review and not dbg.review.get("approve", True):
            print("\nWARNING: Reviewer rejected the patch.", file=sys.stderr)
            return 2
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
