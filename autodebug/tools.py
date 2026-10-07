"""The controlled tool layer. The LLM can only call these functions; there is no shell.

Every call is validated, executed (if needed) in a disposable sandbox copy, clipped, and logged to the
trace. The event id of each call is what the final report cites as evidence.
"""
from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .sandbox import AGENT_TEST_RE, Sandbox, SandboxError, clip, is_protected
from .state import Trace

READONLY_TOOLS = ("list_files", "read_file", "search_code", "run_tests", "run_python", "show_diff")
ALL_TOOLS = READONLY_TOOLS + ("create_test", "apply_edits")

_SELECTOR_RE = re.compile(r"^[\w./:\[\]\-]+$")
_FAILED_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
_NAME_RE = re.compile(r"^[a-z0-9_]{1,40}$")


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: int = 0


def _locate(content: str, old: str) -> tuple[int | None, int | None, str | None, bool]:
    """Find `old` exactly once in `content`. Falls back to a trailing-whitespace-tolerant line match
    (small local models often get trailing spaces / final newlines wrong)."""
    if not old.strip():
        return None, None, "`old` is empty", False
    n = content.count(old)
    if n == 1:
        i = content.index(old)
        return i, i + len(old), None, False
    if n > 1:
        return None, None, f"`old` matches {n} places; include more surrounding lines", False
    lines = content.splitlines(keepends=True)
    want = [l.rstrip() for l in old.strip("\n").splitlines()]
    hits = [i for i in range(len(lines) - len(want) + 1)
            if [l.rstrip() for l in lines[i:i + len(want)]] == want]
    if len(hits) == 1:
        start = sum(len(l) for l in lines[:hits[0]])
        return start, start + sum(len(l) for l in lines[hits[0]:hits[0] + len(want)]), None, True
    if len(hits) > 1:
        return None, None, "`old` matches several places (ignoring trailing whitespace)", False
    return None, None, "`old` not found: copy it EXACTLY from the file, including indentation", False


class Tools:
    def __init__(self, sandbox: Sandbox, trace: Trace, *, test_timeout: float = 120,
                 max_changed_lines: int = 80) -> None:
        self.sb, self.trace = sandbox, trace
        self.test_timeout, self.max_changed_lines = test_timeout, max_changed_lines
        self.files_read: list[str] = []

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: Any) -> ToolResult:
        if name not in ALL_TOOLS:
            res = ToolResult(False, f"unknown tool {name!r}. Available: {', '.join(ALL_TOOLS)}")
        elif not isinstance(args, dict):
            res = ToolResult(False, "args must be a JSON object")
        else:
            try:
                res = getattr(self, f"tool_{name}")(**args)
            except TypeError as e:
                res = ToolResult(False, f"bad arguments for {name}: {e}")
            except (SandboxError, ValueError, OSError) as e:
                res = ToolResult(False, str(e))
        res.event_id = self.trace.emit("tool_call", tool=name, args=clip(str(args), 1500), ok=res.ok,
                                       output=clip(res.output, 6000), data=res.data)
        return res

    # ------------------------------------------------------------------ read-only tools
    def tool_list_files(self, path: str = ".", limit: int = 200) -> ToolResult:
        base = self.sb.resolve(path)
        if not base.is_dir():
            return ToolResult(False, f"not a directory: {path}")
        rels = [r for r in self.sb.files()
                if path in (".", "") or r.startswith(base.relative_to(self.sb.ws.resolve()).as_posix() + "/")]
        shown = rels[:int(limit)]
        more = f"\n... {len(rels) - len(shown)} more" if len(rels) > len(shown) else ""
        return ToolResult(True, "\n".join(shown) + more)

    def tool_read_file(self, path: str, start: int = 1, end: int | None = None) -> ToolResult:
        p = self.sb.resolve(path)
        if not p.is_file():
            return ToolResult(False, f"no such file: {path}")
        if p.stat().st_size > 1_000_000:
            return ToolResult(False, "file too large (>1MB)")
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return ToolResult(False, "binary file")
        start = max(1, int(start))
        end = min(len(lines), int(end) if end else start + 199, start + 249)
        body = "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(start, end + 1))
        if path not in self.files_read:
            self.files_read.append(path)
        return ToolResult(True, f"{path} lines {start}-{end} of {len(lines)}\n{body}")

    def tool_search_code(self, text: str, path: str = ".", limit: int = 40) -> ToolResult:
        """Literal, case-insensitive search (deliberately not a regex: no ReDoS in the orchestrator)."""
        if not isinstance(text, str) or not text.strip():
            return ToolResult(False, "`text` must be a non-empty string")
        needle, hits = text.lower(), []
        prefix = "" if path in (".", "") else self.sb.resolve(path).relative_to(self.sb.ws.resolve()).as_posix()
        for rel in self.sb.files():
            if prefix and not (rel == prefix or rel.startswith(prefix + "/")):
                continue
            p = self.sb.ws / rel
            if p.stat().st_size > 1_000_000:
                continue
            try:
                for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if needle in line.lower():
                        hits.append(f"{rel}:{n}: {line.strip()[:160]}")
            except UnicodeDecodeError:
                continue
            if len(hits) >= int(limit):
                break
        return ToolResult(True, "\n".join(hits[:int(limit)]) or "no matches")

    def tool_show_diff(self) -> ToolResult:
        return ToolResult(True, clip(self.sb.diff(), 6000) or "(no changes)")

    def tool_run_tests(self, selector: str | None = None, runs: int = 1) -> ToolResult:
        argv = [self.sb.python, "-m", "pytest", "-q", "-rf", "--no-header", "--tb=short", "-p", "no:cacheprovider"]
        if selector:
            if not _SELECTOR_RE.match(selector) or selector.startswith("-"):
                return ToolResult(False, "invalid selector; use e.g. tests/test_x.py::test_y")
            if ".." in selector.split("::")[0].split("/"):
                return ToolResult(False, "selector escapes the workspace")
            argv.append(selector)
        runs = max(1, min(int(runs), 50))
        codes, failed_ids, last_fail, last = [], set(), "", ""
        with self.sb.scratch() as d:
            for _ in range(runs):
                r = self.sb.run(argv, d, self.test_timeout)
                codes.append(r.returncode)
                last = r.output
                if "No module named pytest" in last:
                    return ToolResult(False, "pytest is not installed for the interpreter running the sandbox")
                if r.returncode != 0:
                    last_fail = last
                failed_ids |= set(_FAILED_RE.findall(r.output))
        failed_runs = sum(1 for c in codes if c == 1)
        error_runs = sum(1 for c in codes if c not in (0, 1))
        summary = (f"pytest x{runs}: {runs - failed_runs - error_runs} passed, {failed_runs} failed, "
                   f"{error_runs} errored/timeout (return codes: {codes})")
        shown = last_fail or last
        return ToolResult(True, f"{summary}\n{clip(shown, 3500)}",
                          {"runs": runs, "failed_runs": failed_runs, "error_runs": error_runs,
                           "failed_ids": sorted(failed_ids), "codes": codes, "excerpt": clip(shown, 3500)})

    def tool_run_python(self, code: str, timeout: float = 30) -> ToolResult:
        if not isinstance(code, str) or not code.strip() or len(code) > 8000:
            return ToolResult(False, "`code` must be a non-empty string under 8000 chars")
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ToolResult(False, f"experiment has a syntax error: {e}")
        with self.sb.scratch() as d:
            (d / "_autodebug_exp.py").write_text(code, encoding="utf-8")
            r = self.sb.run([self.sb.python, "_autodebug_exp.py"], d, min(float(timeout), 120))
        return ToolResult(True, f"exit code: {r.returncode}\n{clip(r.output, 3500)}",
                          {"returncode": r.returncode, "timed_out": r.timed_out})

    # ------------------------------------------------------------------ mutating tools
    def tool_create_test(self, name: str, code: str) -> ToolResult:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            return ToolResult(False, "`name` must match [a-z0-9_]{1,40}")
        if not isinstance(code, str) or "def test_" not in code or len(code) > 6000:
            return ToolResult(False, "`code` must define at least one `def test_...` and be under 6000 chars")
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ToolResult(False, f"test has a syntax error: {e}")
        rel = f"tests/test_agent_{name}.py"
        target = self.sb.resolve(rel)
        if target.exists():
            return ToolResult(False, f"{rel} already exists; choose another name")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        return ToolResult(True, f"created {rel}", {"path": rel})

    def tool_apply_edits(self, edits: list) -> ToolResult:
        """Atomic search/replace edits. Existing tests are read-only; syntax is checked before writing."""
        if not isinstance(edits, list) or not edits or len(edits) > 10:
            return ToolResult(False, "`edits` must be a list of 1-10 {path, old, new} objects")
        staged: dict[str, str] = {}
        originals: dict[str, str] = {}
        for e in edits:
            if not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in ("path", "old", "new")):
                return ToolResult(False, "each edit needs string fields path, old, new")
            rel = e["path"].strip().removeprefix("./")
            if is_protected(rel) or AGENT_TEST_RE.match(rel):
                return ToolResult(False, f"{rel} is protected (tests/config are read-only); fix the source instead")
            p = self.sb.resolve(rel)
            if not p.is_file():
                return ToolResult(False, f"no such file: {rel}")
            content = staged.get(rel)
            if content is None:
                content = originals[rel] = p.read_text(encoding="utf-8")
            s, t, err, loose = _locate(content, e["old"])
            if err:
                return ToolResult(False, f"{rel}: {err}")
            new = e["new"]
            if loose and content[s:t].endswith("\n") and not new.endswith("\n"):
                new += "\n"
            staged[rel] = content[:s] + new + content[t:]
        changed = 0
        for rel, new_content in staged.items():
            changed += sum(1 for l in difflib.unified_diff(originals[rel].splitlines(), new_content.splitlines(), lineterm="")
                           if l[:1] in "+-" and l[:3] not in ("+++", "---"))
            if rel.endswith(".py"):
                try:
                    ast.parse(new_content)
                except SyntaxError as se:
                    return ToolResult(False, f"{rel}: edit would create a syntax error: {se}")
        already = sum(1 for l in self.sb.diff().splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        total = already + changed
        if total > self.max_changed_lines:
            return ToolResult(False, f"patch too large ({total} changed lines > {self.max_changed_lines}); be more surgical")
        for rel, new_content in staged.items():
            self.sb.resolve(rel).write_text(new_content, encoding="utf-8")
        return ToolResult(True, f"applied {len(edits)} edit(s) to {', '.join(sorted(staged))}",
                          {"files": sorted(staged), "changed_lines": changed})
