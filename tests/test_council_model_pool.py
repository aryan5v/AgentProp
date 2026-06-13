"""Tests for the Council model pool and retrieval tools (no network)."""

from __future__ import annotations

import pytest

from agentprop.council import (
    ModelPool,
    ModelResponse,
    ModelSpec,
    NullRetrieval,
    OpenRouterWebSearch,
)
from agentprop.evaluation.llm_execution import (
    LLMExecutionResult,
    LLMUsage,
    _extract_citations,
)


def _spec(name: str, in_p: float, out_p: float, tier: int = 1, tags=()) -> ModelSpec:
    return ModelSpec(
        name=name,
        input_price_per_mtok=in_p,
        output_price_per_mtok=out_p,
        capability_tier=tier,
        tags=tuple(tags),
    )


class _FakeClient:
    """Records calls and returns a canned result with the given model name."""

    last_kwargs: dict = {}

    def __init__(self, *, model: str, **_: object) -> None:
        self.model = model

    def chat(self, **kwargs: object) -> LLMExecutionResult:
        _FakeClient.last_kwargs = kwargs
        return LLMExecutionResult(
            model=self.model,
            prompt=str(kwargs.get("user_prompt", "")),
            response=f"answer from {self.model}",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_s=0.5,
            citations=("https://example.com/a",),
        )


def _pool(monkeypatch, specs) -> ModelPool:
    pool = ModelPool(specs=tuple(specs), api_key="test-key")
    monkeypatch.setattr(
        "agentprop.council.model_pool.OpenAICompatibleChatClient", _FakeClient
    )
    return pool


def test_cost_usd_from_usage() -> None:
    spec = _spec("m", in_p=1.0, out_p=2.0)
    cost = spec.cost_usd(LLMUsage(prompt_tokens=1_000_000, completion_tokens=500_000))
    assert cost == pytest.approx(1.0 + 1.0)


def test_supports_tier_and_tags() -> None:
    spec = _spec("m", 1, 1, tier=2, tags=("search", "code"))
    assert spec.supports(min_tier=2, required_tags=["search"])
    assert not spec.supports(min_tier=3)
    assert not spec.supports(required_tags=["vision"])


def test_candidates_cheapest_first_and_filtered() -> None:
    pool = ModelPool(
        specs=(
            _spec("cheap", 0.1, 0.2, tier=1, tags=("search",)),
            _spec("mid", 1.0, 2.0, tier=2, tags=("search",)),
            _spec("nosrch", 0.05, 0.05, tier=3, tags=()),
        ),
        api_key="k",
    )
    names = [s.name for s in pool.candidates(required_tags=["search"])]
    assert names == ["cheap", "mid"]
    # min_tier=2 keeps mid+nosrch; ordering is by price, so cheaper nosrch first.
    assert [s.name for s in pool.candidates(min_tier=2)] == ["nosrch", "mid"]


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError):
        ModelPool(specs=(_spec("m", 1, 1), _spec("m", 2, 2)), api_key="k")


def test_call_resolves_cost_and_citations(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("m", in_p=1.0, out_p=2.0)])
    resp = pool.call("m", system_prompt="s", user_prompt="u")
    assert resp.ok
    assert resp.text == "answer from m"
    assert resp.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1_000_000)
    assert resp.citations == ("https://example.com/a",)


def test_call_isolates_failures(monkeypatch) -> None:
    class _Boom(_FakeClient):
        def chat(self, **kwargs: object) -> LLMExecutionResult:
            raise RuntimeError("provider down")

    pool = ModelPool(specs=(_spec("m", 1, 1),), api_key="k")
    monkeypatch.setattr("agentprop.council.model_pool.OpenAICompatibleChatClient", _Boom)
    resp = pool.call("m", system_prompt="s", user_prompt="u")
    assert not resp.ok
    assert "provider down" in (resp.error or "")
    assert resp.cost_usd == 0.0


def test_fan_out_runs_all_models(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("a", 1, 1), _spec("b", 1, 1)])
    out = pool.fan_out(["a", "b"], system_prompt="s", user_prompt="u")
    assert set(out) == {"a", "b"}
    assert all(isinstance(r, ModelResponse) and r.ok for r in out.values())


def test_map_assignments_runs_distinct_prompts(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("a", 1, 1), _spec("b", 1, 1)])
    results = pool.map_assignments(
        [("a", "sysA", "qA"), ("b", "sysB", "qB")]
    )
    assert {r.model for r in results} == {"a", "b"}


def test_extra_body_passed_through(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("a", 1, 1)])
    pool.call("a", system_prompt="s", user_prompt="u", extra_body={"plugins": [{"id": "web"}]})
    assert _FakeClient.last_kwargs["extra_body"] == {"plugins": [{"id": "web"}]}


def test_missing_key_does_not_raise_returns_error(monkeypatch) -> None:
    # call() honors its "never raises" contract even for config errors.
    pool = ModelPool(specs=(_spec("m", 1, 1),), api_key="")
    resp = pool.call("m", system_prompt="s", user_prompt="u")
    assert not resp.ok
    assert "no API key" in (resp.error or "")


def test_unknown_model_returns_error_not_raise(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("m", 1, 1)])
    resp = pool.call("does-not-exist", system_prompt="s", user_prompt="u")
    assert not resp.ok
    assert "KeyError" in (resp.error or "")


def test_fan_out_dedupes_models(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("a", 1, 1)])
    out = pool.fan_out(["a", "a", "a"], system_prompt="s", user_prompt="u")
    assert set(out) == {"a"}


def test_map_assignments_merges_per_item_extra_body(monkeypatch) -> None:
    pool = _pool(monkeypatch, [_spec("a", 1, 1)])
    pool.map_assignments([("a", "sys", "q", {"plugins": [{"id": "web"}]})])
    assert _FakeClient.last_kwargs["extra_body"] == {"plugins": [{"id": "web"}]}


def test_extra_body_cannot_override_reserved_keys() -> None:
    from agentprop.evaluation.llm_execution import OpenAICompatibleChatClient

    client = OpenAICompatibleChatClient(api_key="k", model="m")
    with pytest.raises(ValueError):
        client.chat(system_prompt="s", user_prompt="u", extra_body={"model": "evil"})


def test_extract_citations_ignores_non_list_annotations() -> None:
    assert _extract_citations({"choices": [{"message": {"annotations": 5}}]}) == ()


def test_openrouter_web_search_builds_plugin() -> None:
    result = OpenRouterWebSearch(max_results=3).for_subtask("q")
    assert result.enabled
    assert result.extra_body["plugins"][0]["id"] == "web"
    assert result.extra_body["plugins"][0]["max_results"] == 3


def test_null_retrieval_is_disabled() -> None:
    assert not NullRetrieval().for_subtask("q").enabled


def test_extract_citations_from_annotations() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "x",
                    "annotations": [
                        {"url_citation": {"url": "https://a.com"}},
                        {"url": "https://b.com"},
                        {"url_citation": {"url": "https://a.com"}},
                    ],
                }
            }
        ]
    }
    assert _extract_citations(payload) == ("https://a.com", "https://b.com")
    assert _extract_citations({"choices": []}) == ()
