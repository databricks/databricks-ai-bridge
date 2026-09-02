"""Tests for attaching explicit integrations to an OpenAI Agents SDK agent."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agents import Agent
from databricks.sdk import WorkspaceClient
from databricks_openai.agents import McpServer
from mcp.types import Tool

from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction
from databricks_mason.openai import bind_tools
from databricks_mason.openai import mcp as openai_mcp


def test_bind_tools_owns_new_server_lifecycle_and_returns_an_isolated_clone(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        return []

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)

    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com/")),
    )
    existing_tool = cast(Any, object())
    existing_server = McpServer(
        url="https://custom.example.com/mcp",
        name="existing",
        workspace_client=workspace_client,
    )
    existing_handoff = cast(Any, object())
    existing_input_guardrail = cast(Any, object())
    existing_output_guardrail = cast(Any, object())
    original = Agent(
        name="claims",
        tools=[existing_tool],
        mcp_servers=[existing_server],
        handoffs=cast(Any, [existing_handoff]),
        input_guardrails=[existing_input_guardrail],
        output_guardrails=[existing_output_guardrail],
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                original,
                [MCPService(id="claims_mcp", service="main.claims.service")],
                stack=stack,
                workspace_client=workspace_client,
            )

            assert lifecycle == [
                ("connect", "claims_mcp"),
                ("list", "claims_mcp"),
            ]
            assert bound is not original
            assert bound.tools == [existing_tool]
            assert bound.mcp_servers[0] is existing_server
            assert len(bound.mcp_servers) == 2
            generated = bound.mcp_servers[1]
            assert isinstance(generated, McpServer)
            assert generated.params["url"] == (
                "https://workspace.example.com/ai-gateway/mcp-services/main.claims.service"
            )
            assert generated.workspace_client is workspace_client

            for field in (
                "tools",
                "mcp_servers",
                "handoffs",
                "input_guardrails",
                "output_guardrails",
            ):
                assert getattr(bound, field) is not getattr(original, field)

            assert original.tools == [existing_tool]
            assert original.mcp_servers == [existing_server]
            assert original.handoffs == [existing_handoff]
            assert original.input_guardrails == [existing_input_guardrail]
            assert original.output_guardrails == [existing_output_guardrail]
            return bound

        raise AssertionError("unreachable")

    bound = asyncio.run(run())

    assert lifecycle == [
        ("connect", "claims_mcp"),
        ("list", "claims_mcp"),
        ("cleanup", "claims_mcp"),
    ]
    assert original.mcp_servers == [existing_server]
    assert len(bound.mcp_servers) == 2


def test_bind_tools_materializes_uc_function_and_managed_sandbox(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [
                    UCFunction(id="lookup", function="main.claims.lookup"),
                    Sandbox(
                        id="python",
                        scopes=(Scope.volume("main.claims.files"),),
                    ),
                ],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())
    uc_function, sandbox = (cast(McpServer, server) for server in bound.mcp_servers)

    assert uc_function.name == "lookup"
    assert uc_function.params["url"] == (
        "https://workspace.example.com/api/2.0/mcp/functions/main/claims/lookup"
    )
    assert uc_function.workspace_client is workspace_client
    assert sandbox.name == "python"
    assert sandbox.params["url"] == (
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.sandbox"
    )
    assert sandbox.workspace_client is workspace_client
    assert sandbox.tool_filter == {"allowed_tool_names": ["sandbox", "run_code"]}


def test_sandbox_overrides_incoming_downscope_without_mutating_caller_data(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def call_tool(
        _server: McpServer,
        tool_name: str,
        arguments: dict[str, object] | None,
        **kwargs: object,
    ) -> tuple[str, dict[str, object] | None, dict[str, object]]:
        return tool_name, arguments, kwargs

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "call_tool", call_tool)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    arguments: dict[str, object] = {"code": 'print("hello")'}
    incoming_meta = {
        "downscope": {"volumes": []},
        "trace_id": "request-123",
    }

    async def run() -> tuple[str, dict[str, object] | None, dict[str, object]]:
        async with AsyncExitStack() as stack:
            bound = await bind_tools(
                Agent(name="claims"),
                [
                    Sandbox(
                        id="python",
                        scopes=(Scope.volume("main.claims.files"),),
                    )
                ],
                stack=stack,
                workspace_client=workspace_client,
            )
            return cast(
                Any,
                await bound.mcp_servers[0].call_tool(
                    "sandbox",
                    arguments,
                    meta=incoming_meta,
                ),
            )

    tool_name, forwarded_arguments, forwarded_kwargs = asyncio.run(run())

    assert tool_name == "sandbox"
    assert forwarded_arguments is arguments
    assert forwarded_kwargs["meta"] == {
        "downscope": {
            "volumes": [
                {"name": "main.claims.files", "permission": "read_only"},
            ]
        },
        "trace_id": "request-123",
    }
    assert forwarded_kwargs["meta"] is not incoming_meta
    assert incoming_meta == {
        "downscope": {"volumes": []},
        "trace_id": "request-123",
    }


def test_empty_selection_is_a_credential_free_no_op(monkeypatch):
    def unexpected_workspace_client() -> None:
        raise AssertionError("empty integrations must not resolve workspace credentials")

    monkeypatch.setattr(openai_mcp, "WorkspaceClient", unexpected_workspace_client)
    original = Agent(name="claims", tools=[cast(Any, object())])

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(original, (), stack=stack)

    bound = asyncio.run(run())

    assert bound is not original
    assert bound.tools == original.tools
    assert bound.tools is not original.tools
    assert bound.mcp_servers == []


def test_default_client_uses_runtime_workspace_routing_helper(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    def unexpected_workspace_client() -> None:
        raise AssertionError("bind_tools must use the workspace routing helper")

    routed_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://accounts.example.com")),
    )
    helper_calls = 0

    def routed_workspace_client():
        nonlocal helper_calls
        helper_calls += 1
        return routed_client

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    monkeypatch.setattr(openai_mcp, "WorkspaceClient", unexpected_workspace_client)
    monkeypatch.setattr(
        openai_mcp,
        "_default_workspace_client",
        routed_workspace_client,
        raising=False,
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [MCPService(id="claims_mcp", service="main.claims.service")],
                stack=stack,
            )

    bound = asyncio.run(run())

    assert helper_calls == 1
    assert bound.mcp_servers[0].workspace_client is routed_client


@pytest.mark.parametrize(
    "integration",
    [
        MCPService(id="search", service="system.ai.web_search"),
        Sandbox(id="python", scopes=(Scope.volume("main.claims.files"),)),
        UCFunction(id="lookup", function="main.claims.lookup"),
    ],
)
def test_bind_tools_adds_workspace_routing_header_to_every_generated_transport(
    monkeypatch,
    integration,
):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return []

    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "123456789")
    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://accounts.example.com")),
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims"),
                [integration],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())

    server = cast(McpServer, bound.mcp_servers[0])
    assert server.params["headers"] == {"X-Databricks-Org-Id": "123456789"}


def test_bind_tools_rejects_duplicate_integration_ids_before_resolving_credentials(
    monkeypatch,
):
    def unexpected_workspace_client() -> None:
        raise AssertionError("duplicate names must fail before credentials are resolved")

    monkeypatch.setattr(openai_mcp, "_default_workspace_client", unexpected_workspace_client)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="server name.*duplicate"):
                await bind_tools(
                    Agent(name="claims"),
                    [
                        MCPService(id="duplicate", service="main.claims.first"),
                        MCPService(id="duplicate", service="main.claims.second"),
                    ],
                    stack=stack,
                )

    asyncio.run(run())


def test_bind_tools_rejects_generated_name_matching_existing_server_before_credentials(
    monkeypatch,
):
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    existing = McpServer(
        url="https://custom.example.com/mcp",
        name="duplicate",
        workspace_client=workspace_client,
    )

    def unexpected_workspace_client() -> None:
        raise AssertionError("duplicate names must fail before credentials are resolved")

    monkeypatch.setattr(openai_mcp, "_default_workspace_client", unexpected_workspace_client)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="server name.*duplicate"):
                await bind_tools(
                    Agent(name="claims", mcp_servers=[existing]),
                    [MCPService(id="duplicate", service="main.claims.generated")],
                    stack=stack,
                )

    asyncio.run(run())


def test_bind_tools_eagerly_discovers_and_rejects_duplicate_generated_tools(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        return [Tool(name="lookup_claim", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(
                ValueError,
                match="lookup_claim.*first.*second",
            ):
                await bind_tools(
                    Agent(name="claims"),
                    [
                        MCPService(id="first", service="main.claims.first"),
                        MCPService(id="second", service="main.claims.second"),
                    ],
                    stack=stack,
                    workspace_client=workspace_client,
                )
            assert lifecycle == [
                ("connect", "first"),
                ("list", "first"),
                ("connect", "second"),
                ("list", "second"),
            ]

        assert lifecycle == [
            ("connect", "first"),
            ("list", "first"),
            ("connect", "second"),
            ("list", "second"),
            ("cleanup", "second"),
            ("cleanup", "first"),
        ]

    asyncio.run(run())


def test_bind_tools_defers_existing_dynamic_tool_filters_to_request_context(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        if server is existing:
            raise AssertionError("existing dynamic filters require a request context")
        lifecycle.append(("list", server.name))
        return [Tool(name="generated_tool", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    existing = McpServer(
        url="https://custom.example.com/mcp",
        name="custom",
        workspace_client=workspace_client,
        tool_filter=lambda _context, _tool: True,
    )

    async def run() -> Agent:
        async with AsyncExitStack() as stack:
            return await bind_tools(
                Agent(name="claims", mcp_servers=[existing]),
                [MCPService(id="generated", service="main.claims.generated")],
                stack=stack,
                workspace_client=workspace_client,
            )

    bound = asyncio.run(run())

    assert lifecycle == [
        ("connect", "generated"),
        ("list", "generated"),
        ("cleanup", "generated"),
    ]
    assert bound.mcp_servers[0] is existing


def test_tool_discovery_failure_surfaces_at_bind_and_remains_stack_managed(monkeypatch):
    lifecycle: list[tuple[str, str]] = []

    async def connect(server: McpServer) -> None:
        lifecycle.append(("connect", server.name))

    async def cleanup(server: McpServer) -> None:
        lifecycle.append(("cleanup", server.name))

    async def list_tools(server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        lifecycle.append(("list", server.name))
        raise RuntimeError("tool catalog denied")

    monkeypatch.setattr(McpServer, "connect", connect)
    monkeypatch.setattr(McpServer, "cleanup", cleanup)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="tool catalog denied"):
                await bind_tools(
                    Agent(name="claims"),
                    [MCPService(id="claims", service="main.claims.service")],
                    stack=stack,
                    workspace_client=workspace_client,
                )
            assert lifecycle == [("connect", "claims"), ("list", "claims")]

        assert lifecycle == [
            ("connect", "claims"),
            ("list", "claims"),
            ("cleanup", "claims"),
        ]

    asyncio.run(run())


def test_bind_tools_rejects_generated_name_that_collides_with_local_tool(monkeypatch):
    async def no_op(_server: McpServer) -> None:
        return None

    async def list_tools(_server: McpServer, *_args: object, **_kwargs: object) -> list[Tool]:
        return [Tool(name="lookup_claim", inputSchema={})]

    monkeypatch.setattr(McpServer, "connect", no_op)
    monkeypatch.setattr(McpServer, "cleanup", no_op)
    monkeypatch.setattr(McpServer, "list_tools", list_tools)
    workspace_client = cast(
        WorkspaceClient,
        SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com")),
    )
    agent = Agent(
        name="claims",
        tools=[cast(Any, SimpleNamespace(name="lookup_claim"))],
    )

    async def run() -> None:
        async with AsyncExitStack() as stack:
            with pytest.raises(ValueError, match="lookup_claim.*local agent.*claims"):
                await bind_tools(
                    agent,
                    [MCPService(id="claims", service="main.claims.service")],
                    stack=stack,
                    workspace_client=workspace_client,
                )

    asyncio.run(run())
