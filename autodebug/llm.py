"""Local-LLM client (any OpenAI-compatible server: Ollama, llama.cpp, LM Studio) + a scripted fake."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of messy model output (fences, prose, <think> blocks)."""
    text = _THINK.sub("", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # strict=False tolerates raw newlines inside strings, a very common small-model slip
                return json.loads(text[start:i + 1], strict=False)
    raise ValueError("unterminated JSON object")


class BaseLLM:
    model = "unknown"
    calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def chat_json(self, messages: list[dict[str, str]], retries: int = 2) -> dict[str, Any]:
        msgs = list(messages)
        last = ""
        for _ in range(retries + 1):
            last = self.chat(msgs)
            try:
                obj = extract_json(last)
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
            msgs += [{"role": "assistant", "content": last[:2000]},
                     {"role": "user", "content": "That was not a single valid JSON object. "
                                                 "Reply again with ONLY the JSON object."}]
        raise LLMError("model did not return valid JSON: " + last[:200])


class LocalLLM(BaseLLM):
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", *, temperature: float = 0.2,
                 timeout: float = 600, max_tokens: int = 2048, json_mode: bool = False,
                 api_key: str | None = None) -> None:
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.timeout, self.max_tokens = temperature, timeout, max_tokens
        self.json_mode, self.api_key = json_mode, api_key
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.load(resp)
        except urllib.error.HTTPError as e:
            raise LLMError(f"LLM server returned HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LLMError(f"cannot reach local LLM at {self.base_url}: {e}. "
                           "Is the server running (e.g. `ollama serve`) and the model pulled?") from e
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(payload)[:300]}") from e


class ScriptedLLM(BaseLLM):
    """Deterministic stand-in used for tests and golden-trace replays."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses, self.prompts, self.model, self.calls = list(responses), [], "scripted", 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.prompts.append(messages)
        if not self.responses:
            raise LLMError("scripted LLM exhausted")
        r = self.responses.pop(0)
        return r if isinstance(r, str) else json.dumps(r)
