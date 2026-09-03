"""Unit tests for the OpenAI MCP runtime helpers."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture
def openai_mcp(monkeypatch):
    """Import the optional OpenAI runtime module without installing its dependency extra."""
    databricks_openai = ModuleType("databricks_openai")
    databricks_openai.__path__ = []
    agents = ModuleType("databricks_openai.agents")
    agents.__dict__["McpServer"] = object
    tool_manifest = ModuleType("databricks_mason.runtime.tool_manifest")
    tool_manifest.__dict__["ToolRecord"] = object
    tool_manifest.__dict__["downscope_wire"] = lambda _tool: {}
    tool_manifest.__dict__["load_tools"] = lambda **_kwargs: []
    workspace = ModuleType("databricks_mason.runtime.workspace")
    workspace.__dict__["workspace_client"] = lambda: None
    monkeypatch.setitem(sys.modules, "databricks_openai", databricks_openai)
    monkeypatch.setitem(sys.modules, "databricks_openai.agents", agents)
    monkeypatch.setitem(sys.modules, "databricks_mason.runtime.tool_manifest", tool_manifest)
    monkeypatch.setitem(sys.modules, "databricks_mason.runtime.workspace", workspace)

    sys.modules.pop("databricks_mason.openai.mcp", None)
    module = importlib.import_module("databricks_mason.openai.mcp")
    yield module
    sys.modules.pop("databricks_mason.openai.mcp", None)


class _Server:
    def __init__(
        self,
        name: str,
        *,
        connect_error: Exception | None = None,
        list_tools_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.connect_error = connect_error
        self.list_tools_error = list_tools_error
        self.cache_tools_list = False
        self.list_tools_calls = 0
        self.closed = False

    async def __aenter__(self):
        if self.connect_error is not None:
            raise self.connect_error
        return self

    async def __aexit__(self, *_args) -> None:
        self.closed = True

    async def list_tools(self) -> list:
        self.list_tools_calls += 1
        if self.list_tools_error is not None:
            raise self.list_tools_error
        return []


def test_connected_mcp_servers_drops_server_that_fails_to_connect(openai_mcp, monkeypatch):
    healthy = _Server("healthy")
    broken = _Server("slack", connect_error=PermissionError("HTTP error 403"))

    async def servers(_extra):
        return [healthy, broken]

    monkeypatch.setattr(openai_mcp, "mcp_servers", servers)

    async def run():
        async with openai_mcp.connected_mcp_servers([]) as active:
            assert active == [healthy]
            assert healthy.cache_tools_list is True
            assert healthy.list_tools_calls == 1
            assert healthy.closed is False

        assert healthy.closed is True

    asyncio.run(run())


def test_connected_mcp_servers_drops_server_that_fails_to_list_tools(openai_mcp, monkeypatch):
    healthy = _Server("healthy")
    broken = _Server("slack", list_tools_error=PermissionError("HTTP error 403"))

    async def servers(_extra):
        return [healthy, broken]

    monkeypatch.setattr(openai_mcp, "mcp_servers", servers)

    async def run():
        async with openai_mcp.connected_mcp_servers([]) as active:
            assert active == [healthy]
            assert broken.closed is True

        assert healthy.closed is True

    asyncio.run(run())
