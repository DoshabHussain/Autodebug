import os
import shutil
from pathlib import Path

import pytest

from autodebug.sandbox import Sandbox, SandboxError, is_protected
from autodebug.state import Trace
from autodebug.tools import Tools

REG = Path(__file__).resolve().parents[1] / "cases" / "user_registry" / "repo"


@pytest.fixture
def env(tmp_path):
    sb = Sandbox(REG, backend="auto", timeout=20)
    tools = Tools(sb, Trace(tmp_path / "t.jsonl"))
    yield sb, tools
    sb.cleanup()


def test_original_repo_never_modified(env):
    sb, tools = env
    before = (REG / "registry" / "service.py").read_text()
    tools.call("apply_edits", {"edits": [{"path": "registry/service.py", "old": "import time", "new": "import time  # x"}]})
    assert (REG / "registry" / "service.py").read_text() == before
    assert "# x" in (sb.ws / "registry" / "service.py").read_text()


@pytest.mark.parametrize("bad", ["../etc/passwd", "/etc/passwd", "a/../../x", ""])
def test_path_traversal_blocked(env, bad):
    _, tools = env
    assert not tools.call("read_file", {"path": bad}).ok


def test_symlink_escape_blocked(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text("x=1\n")
    try:
        os.symlink("/etc/hostname", repo / "link.txt")
    except OSError:
        pytest.skip("symlinks not supported/permitted on this platform")
    with Sandbox(repo, timeout=10) as sb:
        tools = Tools(sb, Trace(tmp_path / "t.jsonl"))
        assert not tools.call("read_file", {"path": "link.txt"}).ok


def test_tests_are_read_only_and_create_test_is_scoped(env):
    _, tools = env
    assert is_protected("tests/test_registry.py") and is_protected("conftest.py") and not is_protected("registry/service.py")
    r = tools.call("apply_edits", {"edits": [{"path": "tests/test_registry.py", "old": "def test_count", "new": "def test_x"}]})
    assert not r.ok and "protected" in r.output
    assert tools.call("create_test", {"name": "ok_one", "code": "def test_a():\n    assert True\n"}).ok
    assert not tools.call("create_test", {"name": "ok_one", "code": "def test_a():\n    pass\n"}).ok      # no overwrite
    assert not tools.call("create_test", {"name": "../evil", "code": "def test_a():\n    pass\n"}).ok
    assert not tools.call("apply_edits", {"edits": [{"path": "tests/test_agent_ok_one.py", "old": "True", "new": "False"}]}).ok


def test_edit_validation(env):
    _, tools = env
    p = "registry/service.py"
    assert not tools.call("apply_edits", {"edits": [{"path": p, "old": "nope-not-there", "new": "x"}]}).ok
    assert not tools.call("apply_edits", {"edits": [{"path": p, "old": "self._rows", "new": "x"}]}).ok       # ambiguous
    assert not tools.call("apply_edits", {"edits": [{"path": p, "old": "import time", "new": "import ("}]}).ok  # syntax
    # trailing-whitespace tolerant match succeeds
    assert tools.call("apply_edits", {"edits": [{"path": p, "old": "import time   ", "new": "import time\nimport threading"}]}).ok


def test_run_tests_selector_injection_blocked(env):
    _, tools = env
    for bad in ["--collect-only", "tests/x.py; rm -rf /", "../x.py", "-k foo"]:
        assert not tools.call("run_tests", {"selector": bad}).ok


def test_experiment_cannot_touch_workspace_or_leak_env(env, monkeypatch):
    sb, tools = env
    monkeypatch.setenv("SECRET_API_KEY", "hunter2")
    code = ("import os\nopen('registry/service.py','w').write('# wiped')\n"
            "print('KEY=', os.environ.get('SECRET_API_KEY'))")
    r = tools.call("run_python", {"code": code})
    assert r.ok and "KEY= None" in r.output
    assert "wiped" not in (sb.ws / "registry" / "service.py").read_text()
    assert sb.changed_files() == []


def test_timeout_kills_process_tree(env):
    _, tools = env
    r = tools.call("run_python", {"code": "import time\nwhile True: time.sleep(1)", "timeout": 2})
    assert r.ok and r.data["timed_out"]


@pytest.mark.skipif(not shutil.which("unshare"), reason="needs unshare")
def test_no_network_when_isolated(env):
    sb, tools = env
    if sb.backend == "plain":
        pytest.skip("no network isolation available")
    code = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('NET')\nexcept OSError: print('BLOCKED')"
    assert "BLOCKED" in tools.call("run_python", {"code": code}).output


def test_integrity_check_detects_tampering(env):
    sb, _ = env
    assert sb.integrity_violations() == []
    (sb.ws / "tests" / "test_registry.py").write_text("def test_x(): pass\n")
    assert sb.integrity_violations()
