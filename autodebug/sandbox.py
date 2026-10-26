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

IGNORE_DIRS = {".git", ".hg", "node_modules", "__pycache__",
               ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode"}
MAX_FILES = 20000
AGENT_TEST_RE = re.compile(r"^tests/test_agent_[a-z0-9_]+\.py$")
_PROTECTED_NAMES = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "noxfile.py"}


class SandboxError(RuntimeError):
    pass


def clip(text: str, limit: int = 4000) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars truncated]...\n{text[len(text) - tail:]}"


def is_protected(rel: str) -> bool:
    """Existing tests and test configuration are read-only for the agent."""
    p = PurePosixPath(rel)
    name = p.name
    if name in _PROTECTED_NAMES:
        return True
    parts = p.parts
    # Protect files inside the top-level test folder(s)
    if parts and parts[0] in ("tests", "test"):
        return True
    # Protect root-level test scripts
    if len(parts) == 1 and (name.startswith("test_") or name.endswith("_test.py")):
        return True
    return False


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_venv(dir_path: Path) -> bool:
    return (dir_path / "pyvenv.cfg").exists()


def _walk(root: Path, exclude: frozenset[Path] = frozenset()) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        filtered_dirs = []
        for d in dirnames:
            dp = Path(dirpath) / d
            if d in IGNORE_DIRS:
                continue
            if d in (".venv", "venv", "env") and _is_venv(dp):
                continue
            if dp.resolve() in exclude:
                continue
            filtered_dirs.append(d)
        dirnames[:] = filtered_dirs

        for f in filenames:
            p = Path(dirpath) / f
            if f.endswith(".pyc"):
                continue
            if p.is_symlink():
                try:
                    resolved = p.resolve()
                    ws_resolved = root.resolve()
                    if ws_resolved not in resolved.parents and resolved != ws_resolved:
                        continue  # symlink escapes workspace
                except (OSError, RuntimeError):
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
    except (OSError, subprocess.SubprocessError):
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


def _kill_proc_tree(pid: int) -> None:
    """Kill a process and any descendant processes (BUG-028)."""
    try:
        children_file = Path(f"/proc/{pid}/task/{pid}/children")
        if children_file.is_file():
            for cpid_str in children_file.read_text().split():
                _kill_proc_tree(int(cpid_str))
    except Exception:
        pass
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM  # type: ignore[attr-defined]
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


class Sandbox:
    def __init__(self, source: Path | str, *, backend: str = "auto", python: str | None = None,
                 timeout: float = 60, mem_mb: int = 2048, exclude: Iterable[Path | str] = ()) -> None:
        self.source = Path(source).resolve()
        if not self.source.is_dir():
            raise SandboxError(f"not a directory: {source}")
        self.python = str(Path(python).resolve()) if python else sys.executable
        self.timeout, self.mem_mb = timeout, mem_mb
        self.backend = detect_backend(backend)
        if os.name != "posix" and self.backend not in ("auto", "plain"):
            raise SandboxError(f"requested sandbox backend {backend!r} is not usable on this machine")
        self._exclude = frozenset(Path(p).resolve() for p in exclude)
        
        # Ensure temp directory is cleaned up if initialization fails (BUG-033)
        self.root = Path(tempfile.mkdtemp(prefix="autodebug-")).resolve()
        try:
            self.ws = self.root / "ws"
            (self.root / "home").mkdir()
            self._runs = 0
            if sum(1 for _ in _walk(self.source, self._exclude)) > MAX_FILES:
                raise SandboxError(f"repository has more than {MAX_FILES} files; point at a sub-directory")
            self._copy(self.source, self.ws)
            self._orig = {rel: _sha(self.ws / rel) for rel in _walk(self.ws)}
            self.protected_orig = {rel: h for rel, h in self._orig.items() if is_protected(rel)}
        except Exception:
            shutil.rmtree(self.root, ignore_errors=True)
            raise

    # ------------------------------------------------------------------ copying / paths
    def _copy(self, src: Path, dst: Path, fast_link: bool = False) -> None:
        def ignore(d: str, names: list[str]) -> list[str]:
            dp = Path(d)
            ign: list[str] = []
            for n in names:
                if n in IGNORE_DIRS or n.endswith(".pyc"):
                    ign.append(n)
                elif n in (".venv", "venv", "env") and _is_venv(dp / n):
                    ign.append(n)
                elif (dp / n).resolve() in self._exclude:
                    ign.append(n)
            return ign

        if fast_link:
            def link_or_copy(s: str, d: str) -> None:
                try:
                    os.link(s, d)
                except (OSError, NotImplementedError):
                    shutil.copy2(s, d)
            shutil.copytree(src, dst, symlinks=True, ignore=ignore, copy_function=link_or_copy)
        else:
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
        # Use fast linking to avoid slow copies of large repos (BUG-036)
        self._copy(self.ws, d, fast_link=True)
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _env(self, cwd: Path) -> dict[str, str]:
        home = "/tmp" if self.backend == "bwrap" else str(self.root / "home")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": home,
               "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(cwd), "PYTHONUNBUFFERED": "1"}
        if os.name == "nt":
            for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "PATHEXT", "COMSPEC", "TEMP", "TMP", "USERPROFILE"):
                if k in os.environ:
                    env[k] = os.environ[k]
        return env

    def _python_prefixes(self) -> list[str]:
        prefixes = {sys.prefix, sys.base_prefix, sys.exec_prefix}
        exe = Path(self.python)
        prefixes |= {str(exe.parent.parent), str(exe.resolve().parent.parent)}
        try:
            import site
            prefixes.add(site.getusersitepackages())
        except (AttributeError, ImportError, OSError):
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
        # Raise RLIMIT_AS to 8 GiB so multi-threaded workloads don't run out of virtual address space (BUG-030)
        mem = max(self.mem_mb * 1024 * 1024, 8 * 1024 * 1024 * 1024)

        def apply() -> None:
            limits = [(resource.RLIMIT_CPU, int(timeout) + 5), (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),  # type: ignore[attr-defined]
                      (resource.RLIMIT_NOFILE, 512), (resource.RLIMIT_CORE, 0)]  # type: ignore[attr-defined]
            if hasattr(resource, "RLIMIT_NPROC"):
                # Add RLIMIT_NPROC to contain fork bombs (BUG-029)
                limits.append((resource.RLIMIT_NPROC, 256))  # type: ignore[attr-defined]
            if sys.platform.startswith("linux"):
                limits.append((resource.RLIMIT_AS, mem))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                # Apply data/rss limits on macOS where RLIMIT_AS is unsupported (BUG-031)
                if hasattr(resource, "RLIMIT_DATA"):
                    limits.append((resource.RLIMIT_DATA, mem))  # type: ignore[attr-defined]
                if hasattr(resource, "RLIMIT_RSS"):
                    limits.append((resource.RLIMIT_RSS, mem))  # type: ignore[attr-defined]
            for lim, val in limits:
                try:
                    resource.setrlimit(lim, (val, val))  # type: ignore[attr-defined]
                except (ValueError, OSError):
                    pass
        return apply

    def run(self, argv: list[str], cwd: Path, timeout: float | None = None) -> RunResult:
        timeout = self.timeout if timeout is None else timeout
        cmd = self._wrap(argv, cwd)
        t0 = time.monotonic()
        timed_out = False
        with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
            preexec = self._limits(timeout) if os.name == "posix" else None
            start_sess = True if os.name == "posix" else False
            proc = subprocess.Popen(cmd, cwd=cwd, env=self._env(cwd), stdin=subprocess.DEVNULL,
                                    stdout=fo, stderr=fe, start_new_session=start_sess,
                                    preexec_fn=preexec)  # noqa: PLW1509
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if proc.poll() is None:
                    _kill_proc_tree(proc.pid)
                    try:
                        proc.kill()
                    except OSError:
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
            try:
                return p.read_text(encoding="latin-1")
            except Exception:
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
            # Prevent overwriting changes made by the user while the agent was running (BUG-034)
            if dst.is_file() and rel in self._orig:
                if _sha(dst) != self._orig[rel]:
                    continue
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
