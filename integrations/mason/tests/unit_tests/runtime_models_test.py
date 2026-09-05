"""Unit tests for `databricks_mason.runtime.models.list_ai_gateway_models`."""

from __future__ import annotations

from typing import Any, cast

import pytest

from databricks_mason.runtime import models as models_mod
from databricks_mason.runtime.models import list_ai_gateway_models


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    # Keep retry-exercising tests instant.
    monkeypatch.setattr(models_mod, "_RETRY_DELAY_S", 0)


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
    return list_ai_gateway_models(cast(Any, _Client(pages)))


def test_returns_system_ai_names_sorted_without_prefix():
    result = _list(
        [
            {
                "model_services": [
                    _svc("system.ai.zeta-chat"),
                    _svc("system.ai.alpha-chat"),
                ]
            }
        ]
    )
    assert result == ["system.ai.alpha-chat", "system.ai.zeta-chat"]


def test_drops_embeddings_only_services():
    result = _list(
        [
            {
                "model_services": [
                    _svc("system.ai.claude-sonnet-4-5"),
                    _svc("system.ai.gte-large", api_types=["openai/v1/embeddings"]),
                    _svc("system.ai.no-types", api_types=[]),  # lenient: kept when unknown
                ]
            }
        ]
    )
    assert result == ["system.ai.claude-sonnet-4-5", "system.ai.no-types"]


def test_pages_through_next_page_token():
    pages = [
        {"model_services": [_svc("system.ai.a")], "next_page_token": "1"},
        {"model_services": [_svc("system.ai.b")], "next_page_token": ""},
    ]
    client = _Client(pages)
    result = list_ai_gateway_models(cast(Any, client))
    assert result == ["system.ai.a", "system.ai.b"]
    # First call has no page_token; the second carries the token from page one.
    assert client.api_client.calls[0].get("parent") == "schemas/system.ai"
    assert "page_token" not in client.api_client.calls[0]
    assert client.api_client.calls[1].get("page_token") == "1"


def test_empty_listing():
    assert _list([{"model_services": []}]) == []


def test_propagates_errors_after_retries():
    class _Boom:
        @property
        def api_client(self):
            raise PermissionError("no access")

    with pytest.raises(PermissionError):
        list_ai_gateway_models(cast(Any, _Boom()))


def test_retries_transient_list_error():
    calls = {"n": 0}

    class _FlakyApiClient:
        def do(self, method, path, query=None, body=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient 500")
            return {"model_services": [_svc("system.ai.a")]}

    class _FlakyClient:
        api_client = _FlakyApiClient()

    # First attempt fails, retry succeeds -> the model is discovered, not lost.
    assert list_ai_gateway_models(cast(Any, _FlakyClient())) == ["system.ai.a"]
    assert calls["n"] == 2
