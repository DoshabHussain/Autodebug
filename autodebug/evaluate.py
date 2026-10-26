"""Evaluation harness: run cases in two modes (hypothesis-driven vs baseline) and compare.

    python -m autodebug.evaluate cases/ --model qwen2.5-coder:7b --repeats 3

A case is a directory:   <case>/repo/  <case>/BUG_REPORT.md  [<case>/hidden_tests/*.py]
"Resolved" means: the agent finished DONE *and* the hidden holdout tests (never shown to the agent) pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .agent import Config, Debugger
from .cli import _positive_float, _positive_int
from .llm import LocalLLM
from .sandbox import Sandbox
from .state import State


def holdout_passes(case: Path, dbg: Debugger) -> bool | None:
    hidden = case / "hidden_tests"
    if not hidden.is_dir():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(case / "repo", repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        dbg.sb.export_changes(repo)
        (repo / "tests").mkdir(exist_ok=True)
        for f in hidden.glob("*.py"):
            shutil.copy2(f, repo / "tests" / f"test_hidden_{f.name.removeprefix('test_')}")
        with Sandbox(repo, python=dbg.sb.python, timeout=120) as sb, sb.scratch() as d:
            r = sb.run([sb.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], d, 120)
        return r.returncode == 0


def build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autodebug.evaluate")
    p.add_argument("cases", type=Path)
    p.add_argument("--model", default=os.environ.get("AUTODEBUG_MODEL", "qwen2.5-coder:7b"))
    p.add_argument("--base-url", default=os.environ.get("AUTODEBUG_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--api-key", default=os.environ.get("AUTODEBUG_API_KEY"))
    p.add_argument("--repeats", type=_positive_int, default=1, help="runs per case per mode (LLMs are stochastic)")
    p.add_argument("--sandbox", choices=["auto", "bwrap", "unshare", "plain"], default="auto")
    p.add_argument("--python", help="python interpreter used to run target's tests")
    p.add_argument("--max-llm-calls", type=_positive_int, default=60)
    p.add_argument("--max-seconds", type=_positive_float, default=1800.0)
    p.add_argument("--max-iterations", type=_positive_int, default=3)
    p.add_argument("--out", type=Path, default=Path("autodebug-eval"))
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_eval_parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    # Validate cases directory exists (BUG-075)
    if not a.cases.exists() or not a.cases.is_dir():
        print(f"error: not a directory: {a.cases}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = a.out / stamp
    eval_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = eval_dir / "results.jsonl"

    # Only select cases that have both repo/ and BUG_REPORT.md (BUG-078)
    cases: list[Path] = []
    for c in sorted(a.cases.iterdir()):
        if not c.is_dir():
            continue
        has_repo = (c / "repo").is_dir()
        has_bug = (c / "BUG_REPORT.md").is_file()
        if has_repo and has_bug:
            cases.append(c)
        elif has_repo and not has_bug:
            print(f"warning: skipping {c.name}: missing BUG_REPORT.md", file=sys.stderr)

    if not cases:
        print(f"no valid cases found in {a.cases}", file=sys.stderr)
        return 0

    rows = []
    for case in cases:
        for mode in ("hypothesis", "baseline"):
            for i in range(a.repeats):
                out = eval_dir / f"{case.name}-{mode}-{i}"
                cfg = Config(repo=case / "repo", bug_report=(case / "BUG_REPORT.md").read_text(encoding="utf-8", errors="replace"),
                             out_dir=out.resolve(), baseline=(mode == "baseline"),
                             max_iterations=a.max_iterations, max_llm_calls=a.max_llm_calls,
                             max_seconds=a.max_seconds, sandbox_backend=a.sandbox,
                             python=a.python, verbose=False)
                dbg = Debugger(cfg, LocalLLM(a.model, a.base_url, api_key=a.api_key))
                try:
                    res = dbg.run()
                    held = holdout_passes(case, dbg) if res.status is State.DONE else False
                finally:
                    dbg.sb.cleanup()
                row = {"case": case.name, "mode": mode, "run": i, "status": res.status.value,
                       "resolved": bool(held) if held is not None else res.status is State.DONE,
                       "claimed_done_but_holdout_failed": res.status is State.DONE and held is False,
                       **res.stats}
                rows.append(row)
                
                # Incrementally write results to prevent losing work on crashes (BUG-077)
                with open(jsonl_path, "a", encoding="utf-8") as jfh:
                    jfh.write(json.dumps(row) + "\n")

                print(f"{case.name:<20} {mode:<10} {res.status.value:<17} resolved={row['resolved']} "
                      f"calls={res.stats['llm_calls']} {res.stats['seconds']}s")

    summary: dict[str, dict] = {}
    for mode in ("hypothesis", "baseline"):
        rs = [r for r in rows if r["mode"] == mode]
        if rs:
            summary[mode] = {"runs": len(rs), "resolve_rate": sum(r["resolved"] for r in rs) / len(rs),
                             "false_fix_rate": sum(r["claimed_done_but_holdout_failed"] for r in rs) / len(rs),
                             "avg_llm_calls": sum(r["llm_calls"] for r in rs) / len(rs),
                             "avg_seconds": sum(r["seconds"] for r in rs) / len(rs)}
    (eval_dir / "results.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
