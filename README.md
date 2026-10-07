# AutoDebug: an autonomous debugger on a local LLM (no Docker)

Give it a broken Python repository and a bug report. It **reproduces** the bug with a failing test,
**investigates**, forms **falsifiable hypotheses**, runs **experiments** to confirm or refute them, writes a
**patch**, **verifies** it (stress runs + full suite + untouched tests), has it **reviewed**, and hands you:

`patch.diff` (fix + regression test) · `report.md` (root cause + evidence + verification) · `trace.jsonl` (every step)

Pure standard library. Talks to any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio).

## Quick start

```bash
# 1. a local model (example: Ollama). Give it enough context, 4K is too small for this tool:
ollama pull qwen2.5-coder:7b          # 14b / 32b models do noticeably better
OLLAMA_CONTEXT_LENGTH=16384 ollama serve

# 2. run it on the bundled concurrency bug (the repo is COPIED, never modified in place)
python -m autodebug cases/user_registry/repo --bug-file cases/user_registry/BUG_REPORT.md

# other servers
python -m autodebug REPO --bug "..." --base-url http://localhost:1234/v1 --model <name>   # LM Studio
python -m autodebug REPO --bug "..." --base-url http://localhost:8080/v1 --model <name>   # llama.cpp
```
Target repos need `pytest` available to the interpreter (use `--python /path/to/venv/bin/python` to run the
target's tests with its own environment). Add `--apply` to copy the verified fix into the repo.

## How it works

```
REPRODUCING -> INVESTIGATING -> HYPOTHESIZING -> EXPERIMENTING -> PATCHING -> VERIFYING -> REVIEWING -> DONE
                                     ^__________ refuted hypothesis / failed fix (kept as memory) ____|
any state -> ABSTAINED (honest give-up) | BUDGET_EXHAUSTED
```
* The **orchestrator** decides what happens next; the model only fills in each step (as JSON). Roles are
  states with their own prompts, not separate autonomous agents. That keeps small models on the rails.
* **Reproduction gate:** no fix is attempted without a failing test. If the suite passes, the model must write one
  (threads + barrier for races). If it can't, the run ends `ABSTAINED`. Reproduction is measured over N runs.
* **Hypotheses** carry a *prediction that is logged before the experiment runs*; the verdict is judged only from the
  verbatim output. The report quotes tool output from the trace, never the model's paraphrase.
* **Verification:** repro passes N/N runs, full suite has no new failures, and **existing tests/config are
  byte-identical** (they are read-only for the agent). Failed attempts feed back into the next iteration.
* **Tool layer (no shell):** `list_files read_file search_code run_tests run_python show_diff create_test apply_edits`.
  Paths are jail-checked (incl. symlinks), selectors validated, edits are atomic search/replace with a syntax check and a
  size cap, search is literal (no regex, no ReDoS), `create_test` can only create `tests/test_agent_*.py`.

## The sandbox without Docker (read this)

| Layer | What it does |
|---|---|
| Copy | The agent edits a copy. Your repo is untouched until `--apply`. |
| Disposable run dirs | Tests/experiments run in a fresh copy of the working copy, so untrusted code can't alter it. |
| Process limits | Scrubbed env (no API keys, fake `$HOME`), CPU/memory/file-size/fd rlimits, process-group kill on timeout, output captured to size-limited files. |
| OS backend (auto) | `bwrap` (read-only system, no network, private `/tmp`) > `unshare` (no network) > `plain` (rlimits only). |

`plain` and `unshare` are **not** a filesystem security boundary: a malicious repo could read files your user can read.
Install `bubblewrap` (`apt install bubblewrap`) for real isolation, use `--sandbox bwrap` to *require* it, or run the whole
tool in a VM. Linux/macOS/WSL only. macOS has neither `bwrap` nor `unshare`, so it runs in `plain` mode.
Prompt injection from repo text is mitigated structurally (no shell, protected tests, patch caps) but not eliminated.

## Evaluate it (this is what makes it a portfolio project)

```bash
python -m autodebug.evaluate cases/ --model qwen2.5-coder:7b --repeats 3
```
Runs every case in `cases/` in **hypothesis mode** and **baseline mode** (`--baseline`: same pipeline minus the
hypothesis stage) and reports resolve rate, false-fix rate (agent said DONE but hidden holdout tests fail), calls and time.
Add a case = `cases/<name>/{repo/, BUG_REPORT.md, hidden_tests/}`. Aim for 10+ cases (seeded bugs and, ideally, a
SWE-bench-Lite subset in the same layout) and report failures honestly.

## Tests

```bash
python -m pytest tests -q
```
32 tests: state-machine legality, JSON extraction from messy model output, path/symlink/selector/edit validation,
tamper detection, env scrubbing, timeout kill, no-network, and **full end-to-end runs driven by a scripted LLM**
(success, wrong-fix-then-retry, attempted test-cheating, baseline mode, abstain, budget exhaustion).

## Tuning for small local models
* Context is the #1 problem: raise the server's context (≥16K) and set `--ctx-chars` to ~4 × your context in tokens × 0.6.
* If JSON is unreliable, try `--json-mode`, lower `--temperature`, or a larger/instruct model. Reasoning models'
  `<think>` blocks are stripped automatically.
* Budgets: `--max-llm-calls`, `--max-seconds`, `--max-iterations`. Reports and traces are written even on budget exhaustion.

## Known limitations
* Python + pytest only. No dependency installation: the interpreter you pass must already have the target's deps.
* Not validated against a real model in the development environment (scripted-LLM tests only); expect to tune prompts.
* The `bwrap` backend is implemented but untested there; `unshare` and `plain` are tested.
* "Tests pass" is evidence, not proof. Always read the patch.
