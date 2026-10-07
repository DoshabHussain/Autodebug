"""Evaluation harness: run cases in two modes (hypothesis-driven vs baseline) and compare.

    python -m autodebug.evaluate cases/ --model qwen2.5-coder:7b --repeats 3

A case is a directory:   <case>/repo/  <case>/BUG_REPORT.md  [<case>/hidden_tests/*.py]
"Resolved" means: the agent finished DONE *and* the hidden holdout tests (never shown to the agent) pass.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from .agent import Config, Debugger
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autodebug.evaluate")
    p.add_argument("cases", type=Path)
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--repeats", type=int, default=1, help="runs per case per mode (LLMs are stochastic)")
    p.add_argument("--out", type=Path, default=Path("autodebug-eval"))
    p.add_argument("--max-iterations", type=int, default=3)
    a = p.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for case in sorted(c for c in a.cases.iterdir() if (c / "repo").is_dir()):
        for mode in ("hypothesis", "baseline"):
            for i in range(a.repeats):
                out = a.out / stamp / f"{case.name}-{mode}-{i}"
                cfg = Config(repo=case / "repo", bug_report=(case / "BUG_REPORT.md").read_text(),
                             out_dir=out.resolve(), baseline=(mode == "baseline"),
                             max_iterations=a.max_iterations, verbose=False)
                dbg = Debugger(cfg, LocalLLM(a.model, a.base_url))
                try:
                    res = dbg.run()
                    held = holdout_passes(case, dbg) if res.status is State.DONE else False
                finally:
                    dbg.sb.cleanup()
                rows.append({"case": case.name, "mode": mode, "run": i, "status": res.status.value,
                             "resolved": bool(held) if held is not None else res.status is State.DONE,
                             "claimed_done_but_holdout_failed": res.status is State.DONE and held is False,
                             **res.stats})
                print(f"{case.name:<20} {mode:<10} {res.status.value:<17} resolved={rows[-1]['resolved']} "
                      f"calls={res.stats['llm_calls']} {res.stats['seconds']}s")
    summary: dict[str, dict] = {}
    for mode in ("hypothesis", "baseline"):
        rs = [r for r in rows if r["mode"] == mode]
        if rs:
            summary[mode] = {"runs": len(rs), "resolve_rate": sum(r["resolved"] for r in rs) / len(rs),
                             "false_fix_rate": sum(r["claimed_done_but_holdout_failed"] for r in rs) / len(rs),
                             "avg_llm_calls": sum(r["llm_calls"] for r in rs) / len(rs),
                             "avg_seconds": sum(r["seconds"] for r in rs) / len(rs)}
    (a.out / stamp).mkdir(parents=True, exist_ok=True)
    (a.out / stamp / "results.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
