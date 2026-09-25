"""Unit tests for `databricks_agentkit.runtime.model_services.list_ai_gateway_model_services`."""

from __future__ import annotations

from typing import Any, cast

import pytest

from databricks_agentkit.runtime import model_services as model_services_mod
from databricks_agentkit.runtime.model_services import list_ai_gateway_model_services


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    # Keep retry-exercising tests instant.
    monkeypatch.setattr(model_services_mod, "_RETRY_DELAY_S", 0)


def _svc(name: str, api_types=("openai/v1/chat/completions",)) -> dict:
    return {"name": f"model-services/{name}", "supported_api_types": list(api_types)}


class _ApiClient:
    """Fakes WorkspaceClient.api_client.do for the model-services list route, with paging."""

    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.calls: list[dict] = []

    def do(self, method, path, query=None, body=None):
        assert method == "GET"
        assert path == "/api/2.1/unity-catalog/model-services"
        self.calls.append(query or {})
        # Page by the token we handed out last (index into pages).
        index = int((query or {}).get("page_token") or 0)
        return self._pages[index]


class _Client:
    def __init__(self, pages: list[dict]):
        self.api_client = _ApiClient(pages)


def _list(pages: list[dict]) -> list[str]:
    # cast: the fake stands in for a WorkspaceClient (only .api_client.do is used).
    return list_ai_gateway_model_services(cast(Any, _Client(pages)))


def test_returns_system_ai_names_sorted_without_prefix():
    result = _list(
        [
            {
                "model_services": [
                    _svc("system.ai.llama-4-maverick"),
                    _svc("system.ai.claude-opus-4-8"),
                ]
            }
        ]
    )
    assert result == ["system.ai.claude-opus-4-8", "system.ai.llama-4-maverick"]


def test_drops_embeddings_only_services():
    result = _list(
        [
            {
                "model_services": [
                    _svc("system.ai.claude-sonnet-4-5"),
                    _svc("system.ai.gte-large", api_types=["openai/v1/embeddings"]),
                    _svc("system.ai.llama-4-maverick", api_types=[]),  # lenient: kept when unknown
                ]
            }
        ]
    )
    assert result == ["system.ai.claude-sonnet-4-5", "system.ai.llama-4-maverick"]


def test_pages_through_next_page_token():
    pages = [
        {"model_services": [_svc("system.ai.claude-opus-4-8")], "next_page_token": "1"},
        {"model_services": [_svc("system.ai.llama-4-maverick")], "next_page_token": ""},
    ]
    client = _Client(pages)
    result = list_ai_gateway_model_services(cast(Any, client))
    assert result == ["system.ai.claude-opus-4-8", "system.ai.llama-4-maverick"]
    # First call has no page_token; the second carries the token from page one.
    assert client.api_client.calls[0].get("parent") == "schemas/system.ai"
    assert "page_token" not in client.api_client.calls[0]
    assert client.api_client.calls[1].get("page_token") == "1"


def test_page_size_never_exceeds_api_max():
    # The list API rejects page_size > 100 (InvalidParameterValue); every page request must comply.
    pages = [
        {"model_services": [_svc("system.ai.claude-opus-4-8")], "next_page_token": "1"},
        {"model_services": [_svc("system.ai.llama-4-maverick")], "next_page_token": ""},
    ]
    client = _Client(pages)
    list_ai_gateway_model_services(cast(Any, client))
    assert client.api_client.calls, "expected at least one list request"
    for query in client.api_client.calls:
        assert query["page_size"] <= 100


def test_empty_listing():
    assert _list([{"model_services": []}]) == []


def test_propagates_errors_after_retries():
    class _Boom:
        @property
        def api_client(self):
            raise PermissionError("no access")

    with pytest.raises(PermissionError):
        list_ai_gateway_model_services(cast(Any, _Boom()))


def test_retries_transient_list_error():
    calls = {"n": 0}

    class _FlakyApiClient:
        def do(self, method, path, query=None, body=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient 500")
            return {"model_services": [_svc("system.ai.claude-opus-4-8")]}

    class _FlakyClient:
        api_client = _FlakyApiClient()

    # First attempt fails, retry succeeds -> the model is discovered, not lost.
    assert list_ai_gateway_model_services(cast(Any, _FlakyClient())) == [
        "system.ai.claude-opus-4-8"
    ]
    assert calls["n"] == 2
