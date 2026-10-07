"""Docker-free sandbox.

Layers (strongest available is chosen automatically):
  1. The agent's working copy (`ws`) is a COPY of the repo. Nothing executes inside it.
  2. Every run (tests, experiments) happens in a throw-away copy of `ws`, so untrusted code can
     never alter the working copy or the original repo.
  3. Scrubbed environment (no API keys, fake HOME), rlimits (CPU, memory, file size, fds),
     own process group killed on timeout, output captured to size-limited files.
  4. OS isolation backend:  bwrap (bubblewrap: read-only system, no network, private /tmp)
                            > unshare (new network namespace, no network)
                            > plain (rlimits only; NOT a security boundary).

Honest limits: without bwrap the process can still read files your user can read. Do not point this
at repositories you do not trust unless the backend is `bwrap` (or run it inside a VM).
"""
from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

IGNORE_DIRS = {".git", ".hg", ".venv", "venv", "env", "node_modules", "__pycache__",
               ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode"}
MAX_FILES = 20000
AGENT_TEST_RE = re.compile(r"^tests/test_agent_[a-z0-9_]+\.py$")
_PROTECTED_NAMES = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "noxfile.py"}


class SandboxError(RuntimeError):
    pass


def clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    return f"{text[:head]}\n...[{len(text) - limit} chars truncated]...\n{text[-(limit - head):]}"


def is_protected(rel: str) -> bool:
    """Existing tests and test configuration are read-only for the agent."""
    p = PurePosixPath(rel)
    name = p.name
    if name in _PROTECTED_NAMES or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in ("tests", "test", "testing") for part in p.parts[:-1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(root: Path, exclude: frozenset[Path] = frozenset()) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in IGNORE_DIRS and (Path(dirpath) / d).resolve() not in exclude]
        for f in filenames:
            p = Path(dirpath) / f
            if p.is_symlink() or f.endswith(".pyc"):
                continue
            yield p.relative_to(root).as_posix()


@dataclass
class RunResult:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float

    @property
    def output(self) -> str:
        parts = [self.stdout.rstrip(), self.stderr.rstrip()]
        text = "\n".join(p for p in parts if p)
        return text + ("\n[timed out]" if self.timed_out else "")


def _probe(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def detect_backend(pref: str = "auto") -> str:
    if pref not in ("auto", "bwrap", "unshare", "plain"):
        raise SandboxError(f"unknown sandbox backend {pref!r}")
    if pref in ("auto", "bwrap") and shutil.which("bwrap") and _probe(
            ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-net", "true"]):
        return "bwrap"
    if pref in ("auto", "unshare") and shutil.which("unshare") and _probe(
            ["unshare", "--net", "--map-root-user", "true"]):
        return "unshare"
    if pref in ("bwrap", "unshare"):
        raise SandboxError(f"requested sandbox backend {pref!r} is not usable on this machine")
    return "plain"


def _read_clipped(f, limit: int = 20000) -> str:
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size <= limit:
        return f.read().decode("utf-8", "replace")
    head = f.read(limit // 2)
    f.seek(size - limit // 2)
    tail = f.read()
    return (head.decode("utf-8", "replace") + f"\n...[{size - limit} bytes truncated]...\n"
            + tail.decode("utf-8", "replace"))


class Sandbox:
    def __init__(self, source: Path | str, *, backend: str = "auto", python: str | None = None,
                 timeout: float = 60, mem_mb: int = 2048, exclude: Iterable[Path | str] = ()) -> None:
        if os.name != "posix":
            raise SandboxError("POSIX only (Linux, macOS or WSL)")
        self.source = Path(source).resolve()
        if not self.source.is_dir():
            raise SandboxError(f"not a directory: {source}")
        self.python = python or sys.executable
        self.timeout, self.mem_mb = timeout, mem_mb
        self.backend = detect_backend(backend)
        self._exclude = frozenset(Path(p).resolve() for p in exclude)
        self.root = Path(tempfile.mkdtemp(prefix="autodebug-")).resolve()
        self.ws = self.root / "ws"
        (self.root / "home").mkdir()
        self._runs = 0
        if sum(1 for _ in _walk(self.source, self._exclude)) > MAX_FILES:
            raise SandboxError(f"repository has more than {MAX_FILES} files; point at a sub-directory")
        self._copy(self.source, self.ws)
        self._orig = {rel: _sha(self.ws / rel) for rel in _walk(self.ws)}
        self.protected_orig = {rel: h for rel, h in self._orig.items() if is_protected(rel)}

    # ------------------------------------------------------------------ copying / paths
    def _copy(self, src: Path, dst: Path) -> None:
        def ignore(d: str, names: list[str]) -> list[str]:
            return [n for n in names if n in IGNORE_DIRS or n.endswith(".pyc")
                    or (Path(d) / n).resolve() in self._exclude]
        shutil.copytree(src, dst, symlinks=True, ignore=ignore)

    def resolve(self, rel: str) -> Path:
        """Map an agent-supplied relative path to a real path, refusing anything outside `ws`."""
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise SandboxError("invalid path")
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise SandboxError(f"path escapes the workspace: {rel}")
        full = (self.ws / p).resolve()
        ws = self.ws.resolve()
        if full != ws and ws not in full.parents:
            raise SandboxError(f"path escapes the workspace (symlink?): {rel}")
        return full

    def files(self) -> list[str]:
        return sorted(_walk(self.ws))

    # ------------------------------------------------------------------ execution
    @contextmanager
    def scratch(self) -> Iterator[Path]:
        """A disposable copy of the working copy. Everything untrusted runs here."""
        self._runs += 1
        d = self.root / f"run{self._runs}"
        self._copy(self.ws, d)
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _env(self, cwd: Path) -> dict[str, str]:
        home = "/tmp" if self.backend == "bwrap" else str(self.root / "home")
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": home,
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(cwd), "PYTHONUNBUFFERED": "1"}

    def _python_prefixes(self) -> list[str]:
        prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix}
        exe = Path(self.python)
        prefixes |= {str(exe.parent.parent), str(exe.resolve().parent.parent)}
        try:
            import site
            prefixes.add(site.getusersitepackages())
        except Exception:
            pass
        return sorted(p for p in prefixes if p and os.path.exists(p))

    def _wrap(self, argv: list[str], cwd: Path) -> list[str]:
        if self.backend == "unshare":
            return ["unshare", "--net", "--map-root-user", "--", *argv]
        if self.backend != "bwrap":
            return argv
        cmd = ["bwrap", "--die-with-parent", "--unshare-all", "--new-session"]
        bound: list[str] = []
        for p in ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64"):
            if os.path.islink(p):
                cmd += ["--symlink", os.readlink(p), p]
            elif os.path.exists(p):
                cmd += ["--ro-bind", p, p]
                bound.append(p)
        for p in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/alternatives",
                  "/etc/localtime"):
            if os.path.exists(p):
                cmd += ["--ro-bind", p, p]
        for p in self._python_prefixes():
            if not any(p == b or p.startswith(b + "/") for b in bound):
                cmd += ["--ro-bind", p, p]
        cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--bind", str(cwd), str(cwd), "--chdir", str(cwd)]
        return cmd + ["--", *argv]

    def _limits(self, timeout: float):
        if resource is None:
            return None
        mem = self.mem_mb * 1024 * 1024

        def apply() -> None:
            limits = [(resource.RLIMIT_CPU, int(timeout) + 5), (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
                      (resource.RLIMIT_NOFILE, 512), (resource.RLIMIT_CORE, 0)]
            if sys.platform.startswith("linux"):
                limits.append((resource.RLIMIT_AS, mem))
            for lim, val in limits:
                try:
                    resource.setrlimit(lim, (val, val))
                except (ValueError, OSError):
                    pass
        return apply

    def run(self, argv: list[str], cwd: Path, timeout: float | None = None) -> RunResult:
        timeout = timeout or self.timeout
        cmd = self._wrap(argv, cwd)
        t0 = time.monotonic()
        timed_out = False
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            proc = subprocess.Popen(cmd, cwd=cwd, env=self._env(cwd), stdin=subprocess.DEVNULL,
                                    stdout=fo, stderr=fe, start_new_session=True,
                                    preexec_fn=self._limits(timeout))
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:                      # also reaps any stray children in the process group
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait()
            out, err = _read_clipped(fo), _read_clipped(fe)
        return RunResult(argv, None if timed_out else proc.returncode, out, err, timed_out,
                         time.monotonic() - t0)

    # ------------------------------------------------------------------ change tracking
    def changed_files(self) -> list[str]:
        now = {rel: _sha(self.ws / rel) for rel in _walk(self.ws)}
        return sorted(rel for rel in set(now) | set(self._orig) if now.get(rel) != self._orig.get(rel))

    def _read(self, base: Path, rel: str) -> str | None:
        p = base / rel
        if not p.is_file():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "<binary>"

    def diff(self) -> str:
        out: list[str] = []
        for rel in self.changed_files():
            a, b = self._read(self.source, rel), self._read(self.ws, rel)
            out.extend(difflib.unified_diff(
                (a or "").splitlines(True), (b or "").splitlines(True),
                fromfile=f"a/{rel}" if a is not None else "/dev/null",
                tofile=f"b/{rel}" if b is not None else "/dev/null"))
            if out and not out[-1].endswith("\n"):
                out.append("\n")
        return "".join(out)

    def integrity_violations(self) -> list[str]:
        """Existing tests/config must be untouched; the only new test files allowed are agent repro tests."""
        bad = []
        for rel, h in self.protected_orig.items():
            p = self.ws / rel
            if not p.is_file() or _sha(p) != h:
                bad.append(f"protected file modified or deleted: {rel}")
        for rel in _walk(self.ws):
            if is_protected(rel) and rel not in self.protected_orig and not AGENT_TEST_RE.match(rel):
                bad.append(f"unexpected new protected file: {rel}")
        return bad

    def revert_changes(self, keep_agent_tests: bool = True) -> None:
        for rel in self.changed_files():
            if keep_agent_tests and AGENT_TEST_RE.match(rel):
                continue
            dst = self.ws / rel
            src = self.source / rel
            if rel in self._orig and src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.exists():
                dst.unlink()

    def export_changes(self, dest: Path | str) -> list[str]:
        """Copy changed/new files from the working copy into `dest` (e.g. the real repo)."""
        dest = Path(dest)
        done = []
        for rel in self.changed_files():
            src, dst = self.ws / rel, dest / rel
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                done.append(rel)
            elif dst.exists():
                dst.unlink()
                done.append(rel)
        return done

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
