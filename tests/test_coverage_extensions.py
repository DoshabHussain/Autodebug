"""Tests addressing test-suite gaps (BUG-099 to BUG-103) and verifying defect fixes."""
from __future__ import annotations

import http.server
import json
import socketserver
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from autodebug import __version__
from autodebug.agent import Config, Debugger
from autodebug.cli import build_parser, main as cli_main
from autodebug.evaluate import build_eval_parser, main as eval_main
from autodebug.llm import LLMError, LocalLLM, ScriptedLLM, extract_json
from autodebug.sandbox import Sandbox, SandboxError, clip, detect_backend, is_protected
from autodebug.state import State, Trace
from autodebug.tools import Tools


# =====================================================================
# BUG-102: Unit tests for list_files, read_file, search_code (tools.py)
# =====================================================================
def test_tools_list_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
    sub = repo / "sub"
    sub.mkdir()
    (sub / "util.py").write_text("y = 2\n", encoding="utf-8")

    with Sandbox(repo, timeout=10) as sb:
        tools = Tools(sb, Trace(tmp_path / "t.jsonl"))

        # BUG-001: path='.' and path='./' must list identical files
        r_dot = tools.tool_list_files(".")
        r_slash = tools.tool_list_files("./")
        assert r_dot.ok and r_slash.ok
        assert r_dot.output == r_slash.output
        assert "mod.py" in r_dot.output
        assert "sub/util.py" in r_dot.output

        # subfolder listing
        r_sub = tools.tool_list_files("sub")
        assert r_sub.ok and "sub/util.py" in r_sub.output

        # non-existent folder
        r_non = tools.tool_list_files("nonexistent")
        assert not r_non.ok and "not a directory" in r_non.output


def test_tools_read_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (repo / "empty.py").write_text("", encoding="utf-8")

    with Sandbox(repo, timeout=10) as sb:
        tools = Tools(sb, Trace(tmp_path / "t.jsonl"))

        # Normal read
        r = tools.tool_read_file("f.py", 1, 2)
        assert r.ok and "line1" in r.output and "line2" in r.output

        # BUG-002: out-of-range & inverted line numbers
        assert not tools.tool_read_file("f.py", start=50).ok
        assert not tools.tool_read_file("f.py", start=3, end=1).ok
        assert not tools.tool_read_file("f.py", start=1, end=-5).ok
        assert not tools.tool_read_file("f.py", start=-1).ok

        # BUG-003: empty file
        r_emp = tools.tool_read_file("empty.py")
        assert r_emp.ok and "(empty file)" in r_emp.output

        # BUG-005: files_read normalization
        tools.tool_read_file("f.py")
        tools.tool_read_file("./f.py")
        assert tools.files_read.count("f.py") == 1

        # BUG-019: non-integer / NaN / Inf rejected
        assert not tools.call("read_file", {"path": "f.py", "start": float("inf")}).ok
        assert not tools.call("read_file", {"path": "f.py", "start": float("nan")}).ok


def test_tools_search_code(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def target_fn():\n    return 42\n", encoding="utf-8")
    (repo / "b.py").write_text("target_fn()\n", encoding="utf-8")

    with Sandbox(repo, timeout=10) as sb:
        tools = Tools(sb, Trace(tmp_path / "t.jsonl"))

        # Valid search
        r = tools.tool_search_code("target_fn")
        assert r.ok and "a.py:1:" in r.output and "b.py:1:" in r.output

        # BUG-006: non-existent directory returns error
        r_bad = tools.tool_search_code("target_fn", path="missing_dir")
        assert not r_bad.ok and "no such path" in r_bad.output

        # BUG-007: limit < 1 returns error
        assert not tools.tool_search_code("target_fn", limit=0).ok
        assert not tools.tool_search_code("target_fn", limit=-2).ok


def test_tools_pep263_latin1(tmp_path):
    # BUG-009: valid PEP 263 Latin-1 source file
    repo = tmp_path / "repo"
    repo.mkdir()
    latin1_code = "# -*- coding: latin-1 -*-\n# café\nx = 1\n".encode("latin-1")
    (repo / "latin.py").write_bytes(latin1_code)

    with Sandbox(repo, timeout=10) as sb:
        tools = Tools(sb, Trace(tmp_path / "t.jsonl"))
        r = tools.tool_read_file("latin.py")
        assert r.ok
        assert "café" in r.output or "caf" in r.output


# =====================================================================
# BUG-101: LocalLLM HTTP error paths & mock server
# =====================================================================
def test_localllm_http_errors():
    import http.client
    import io
    import urllib.error
    from unittest.mock import MagicMock, patch

    # BUG-041: scheme validation
    with pytest.raises(LLMError, match="base_url scheme must be http or https"):
        LocalLLM("test", "file:///etc/passwd")

    llm = LocalLLM("test", "http://localhost:11434/v1")

    # BUG-037: non-JSON body raises LLMError
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(LLMError, match="LLM server returned invalid JSON"):
            llm.chat([{"role": "user", "content": "hi"}])

    # HTTP error raises LLMError
    err = urllib.error.HTTPError("http://localhost:11434/v1/chat/completions", 500, "Server Error", {}, io.BytesIO(b"internal error"))
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(LLMError, match="HTTP 500"):
            llm.chat([{"role": "user", "content": "hi"}])

    # BUG-039: content as list of parts
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": [{"type": "text", "text": '{"result": 123}'}]}}]
    }).encode()
    with patch("urllib.request.urlopen", return_value=mock_resp):
        resp = llm.chat([{"role": "user", "content": "hi"}])
        assert '{"result": 123}' in resp

    # BUG-038: IncompleteRead raises LLMError
    with patch("urllib.request.urlopen", side_effect=http.client.IncompleteRead(b"abc")):
        with pytest.raises(LLMError, match="cannot reach local LLM"):
            llm.chat([{"role": "user", "content": "hi"}])

    # Normal response
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": '{"ok": true}'}}]
    }).encode()
    with patch("urllib.request.urlopen", return_value=mock_resp):
        resp_ok = llm.chat([{"role": "user", "content": "hi"}])
        assert '{"ok": true}' in resp_ok


# =====================================================================
# BUG-103: Sandbox backend wrappers (command construction)
# =====================================================================
def test_sandbox_backend_wrappers(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "m.py").write_text("x = 1\n", encoding="utf-8")

    with Sandbox(repo, backend="plain", timeout=10) as sb:
        # plain backend returns unmodified command
        cmd = ["python", "-m", "pytest"]
        assert sb._wrap(cmd, sb.ws) == cmd

        # unshare backend wraps with unshare
        sb.backend = "unshare"
        wrapped_unshare = sb._wrap(cmd, sb.ws)
        assert wrapped_unshare == ["unshare", "--net", "--map-root-user", "--", *cmd]

        # bwrap backend wraps with bwrap args
        sb.backend = "bwrap"
        wrapped_bwrap = sb._wrap(cmd, sb.ws)
        assert wrapped_bwrap[0] == "bwrap"
        assert "--die-with-parent" in wrapped_bwrap
        assert "--unshare-all" in wrapped_bwrap
        assert wrapped_bwrap[-len(cmd):] == cmd

    assert detect_backend("plain") == "plain"
    with pytest.raises(SandboxError, match="unknown sandbox backend"):
        detect_backend("invalid_backend_name")


# =====================================================================
# BUG-099: cli.py and __main__.py tests
# =====================================================================
def test_cli_argument_parsing(tmp_path):
    p = build_parser()
    # BUG-070: positive validators
    with pytest.raises(SystemExit):
        p.parse_args([str(tmp_path), "--bug", "x", "--max-iterations", "-1"])
    with pytest.raises(SystemExit):
        p.parse_args([str(tmp_path), "--bug", "x", "--max-seconds", "0"])

    # BUG-042, BUG-043, BUG-044, BUG-073, BUG-074: new CLI options
    args = p.parse_args([
        str(tmp_path), "--bug", "hello",
        "--api-key", "secret-key",
        "--timeout", "45",
        "--max-tokens", "4096",
        "--test-timeout", "150",
        "--max-changed-lines", "50",
    ])
    assert args.api_key == "secret-key"
    assert args.timeout == 45.0
    assert args.max_tokens == 4096
    assert args.test_timeout == 150.0
    assert args.max_changed_lines == 50


def test_cli_validation_paths(tmp_path, capsys):
    # BUG-064: non-existent repo
    code = cli_main(["/no/such/repository", "--bug", "test"])
    assert code == 2
    err = capsys.readouterr().err
    assert "not a directory" in err

    # BUG-066: missing bug file
    repo = tmp_path / "repo"
    repo.mkdir()
    code = cli_main([str(repo), "--bug-file", str(tmp_path / "missing_bug.md")])
    assert code == 2
    assert "no such bug file" in capsys.readouterr().err

    # BUG-069: empty bug
    code = cli_main([str(repo), "--bug", "   "])
    assert code == 2
    assert "bug description required" in capsys.readouterr().err


def test_main_module():
    res = subprocess.run([sys.executable, "-m", "autodebug", "--version"],
                         capture_output=True, text=True)
    assert res.returncode == 0
    assert f"autodebug {__version__}" in res.stdout or f"autodebug {__version__}" in res.stderr


# =====================================================================
# BUG-100: evaluate.py tests
# =====================================================================
def test_evaluate_cli(tmp_path, capsys):
    # BUG-075: missing cases dir
    code = eval_main(["/no/such/cases/dir"])
    assert code == 2
    assert "not a directory" in capsys.readouterr().err

    # BUG-076: repeats 0 rejected
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    code = eval_main([str(cases_dir), "--repeats", "0"])
    assert code == 2

    # BUG-078: case without BUG_REPORT.md skipped with warning
    bad_case = cases_dir / "bad_case"
    bad_case.mkdir()
    (bad_case / "repo").mkdir()
    code = eval_main([str(cases_dir)])
    assert code == 0
    assert "skipping bad_case: missing BUG_REPORT.md" in capsys.readouterr().err
