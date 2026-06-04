import json
from typing import Any

import agentprop.evaluation.llm_execution as llm_execution
from agentprop.evaluation import OpenAICompatibleChatClient, openai_compatible_env_status


class _FakeHTTPResponse:
    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "Final answer with pytest."}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            }
        ).encode("utf-8")


def test_openai_compatible_chat_client_parses_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse()

    monkeypatch.setattr(llm_execution.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleChatClient(
        api_key="test-key",
        model="test-model",
        base_url="https://router.example/v1",
        timeout_s=3,
    )

    result = client.chat(system_prompt="system", user_prompt="user", max_tokens=100)

    assert captured["url"] == "https://router.example/v1/chat/completions"
    assert captured["timeout"] == 3
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["max_tokens"] == 100
    assert result.response == "Final answer with pytest."
    assert result.usage.total_tokens == 19


def test_openai_compatible_env_status_reports_missing_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    status = openai_compatible_env_status()

    assert not status["ready"]
    assert "OPENAI_API_KEY" in status["missing"]


def test_openai_compatible_env_status_accepts_openai_compatible_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "router-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://router.example/v1")

    status = openai_compatible_env_status()

    assert status["ready"]
    assert status["api_key_env"] == "OPENAI_API_KEY"
    assert status["model"] == "router-model"
