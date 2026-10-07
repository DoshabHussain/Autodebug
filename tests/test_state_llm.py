import pytest

from autodebug.llm import LLMError, ScriptedLLM, extract_json
from autodebug.state import Budget, BudgetExceeded, IllegalTransition, State, StateMachine


def test_legal_path():
    sm = StateMachine()
    for s in (State.REPRODUCING, State.INVESTIGATING, State.HYPOTHESIZING, State.EXPERIMENTING,
              State.PATCHING, State.VERIFYING, State.REVIEWING, State.DONE):
        sm.to(s)


@pytest.mark.parametrize("frm,to", [(State.INITIAL, State.PATCHING), (State.REPRODUCING, State.DONE),
                                    (State.PATCHING, State.DONE), (State.VERIFYING, State.PATCHING)])
def test_illegal_transitions(frm, to):
    sm = StateMachine()
    sm.state = frm
    with pytest.raises(IllegalTransition):
        sm.to(to)


def test_terminal_is_final_and_budget_reachable_anywhere():
    sm = StateMachine()
    sm.state = State.PATCHING
    sm.to(State.BUDGET_EXHAUSTED)
    with pytest.raises(IllegalTransition):
        sm.to(State.HYPOTHESIZING)


def test_budget():
    b = Budget(max_llm_calls=2)
    b.charge(2)
    with pytest.raises(BudgetExceeded):
        b.check()


@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    'Sure!\n```json\n{"a": 1}\n```\nhope that helps',
    '<think>hmm {not json}</think>{"a": 1}',
    '{"a": 1} trailing',
    '{"code": "line1\nline2", "a": 1}',          # raw newline inside a string
])
def test_extract_json(raw):
    assert extract_json(raw)["a"] == 1 or "code" in extract_json(raw)


def test_chat_json_repairs_then_gives_up():
    llm = ScriptedLLM(["not json", '{"ok": true}'])
    assert llm.chat_json([{"role": "user", "content": "x"}]) == {"ok": True}
    with pytest.raises(LLMError):
        ScriptedLLM(["bad", "bad", "bad"]).chat_json([{"role": "user", "content": "x"}])
