"""Unit tests for `databricks_mason.runtime.models.list_chat_model_endpoints`."""

from __future__ import annotations

from typing import Any, cast

import pytest

from databricks_mason.runtime.models import list_chat_model_endpoints


class _Endpoint:
    def __init__(self, name, task):
        self.name = name
        self.task = task


class _ServingEndpoints:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def list(self):
        return iter(self._endpoints)


class _Client:
    def __init__(self, endpoints):
        self.serving_endpoints = _ServingEndpoints(endpoints)


def _list(endpoints) -> list[str]:
    # cast: the fake stands in for a WorkspaceClient (only .serving_endpoints.list() is used).
    return list_chat_model_endpoints(cast(Any, _Client(endpoints)))


def test_keeps_only_chat_tasks_sorted():
    assert _list(
        [
            _Endpoint("zeta-chat", "llm/v1/chat"),
            _Endpoint("embeddings", "llm/v1/embeddings"),
            _Endpoint("alpha-chat", "agent/v2/chat"),
            _Endpoint("completions", "llm/v1/completions"),
        ]
    ) == ["alpha-chat", "zeta-chat"]


def test_skips_endpoints_without_task_or_name():
    assert _list(
        [
            _Endpoint("no-task", None),
            _Endpoint(None, "llm/v1/chat"),
            _Endpoint("good", "LLM/V1/CHAT"),  # task match is case-insensitive
        ]
    ) == ["good"]


def test_empty_when_no_chat_endpoints():
    assert _list([_Endpoint("embeddings", "llm/v1/embeddings")]) == []


def test_propagates_list_errors():
    class _Boom:
        @property
        def serving_endpoints(self):
            raise PermissionError("no access")

    with pytest.raises(PermissionError):
        list_chat_model_endpoints(cast(Any, _Boom()))
