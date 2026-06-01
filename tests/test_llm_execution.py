import json
from typing import Any

import agentprop.evaluation.llm_execution as llm_execution
from agentprop.evaluation import OpenAICompatibleChatClient


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
