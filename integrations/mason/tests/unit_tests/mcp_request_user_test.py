"""Request-scoped identity at the framework MCP boundary, without live credentials."""

import asyncio
import builtins
import importlib
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


class FakeServer:
    def __init__(self, name, url, workspace_client=None, **kwargs):
        self.name = name
        self.url = url
        self.workspace_client = workspace_client
        self.kwargs = kwargs

    @classmethod
    def from_uc_function(cls, catalog, schema, function_name, **kwargs):
        return cls(url=f"https://workspace/functions/{catalog}/{schema}/{function_name}", **kwargs)

    def to_connection_dict(self):
        return {"workspace_client": self.workspace_client}

    async def connect(self):
        return self.workspace_client

    async def list_tools(self, *args, **kwargs):
        return []

    async def call_tool(self, *args, **kwargs):
        return SimpleNamespace(isError=False)


@pytest.fixture(params=["langgraph", "openai"])
def adapter(request, monkeypatch):
    module_name = f"databricks_mason.{request.param}.mcp"
    package_name = (
        "databricks_langchain" if request.param == "langgraph" else "databricks_openai.agents"
    )
    package = ModuleType(package_name)
    package.__dict__["DatabricksMCPServer"] = FakeServer
    package.__dict__["McpServer"] = FakeServer

    class FakeClient:
        def __init__(self, servers, **kwargs):
            self.servers = servers
            self.interceptors = kwargs.get("tool_interceptors", [])
            self.handle_tool_errors = kwargs.get("handle_tool_errors", True)

        async def get_tools(self, server_name=None):
            return [
                server
                for server in self.servers
                if server_name is None or server.name == server_name
            ]

    package.__dict__["DatabricksMultiServerMCPClient"] = FakeClient
    monkeypatch.setitem(sys.modules, package_name, package)
    sessions = ModuleType("langchain_mcp_adapters.sessions")
    sessions.__dict__["create_session"] = Mock()
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.sessions", sessions)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    default = SimpleNamespace(config=SimpleNamespace(host="https://workspace"))
    monkeypatch.setattr(module, "workspace_client", Mock(return_value=default))
    monkeypatch.setattr(module, "load_tools", Mock(return_value=[]))
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    yield module
    sys.modules.pop(module_name, None)


def tool(auth=None, kind="mcp", name="search"):
    return SimpleNamespace(
        id=name,
        kind=kind,
        auth=auth,
        service="system.ai.search",
        function="main.tools.lookup",
        downscope=(),
    )


def test_clients_are_selected_per_tool_and_request(adapter):
    alice, bob, app = [
        SimpleNamespace(config=SimpleNamespace(host="https://workspace")) for _ in range(3)
    ]
    for user in (alice, bob):
        resolver = Mock(side_effect=lambda mode, user=user: user if mode == "user" else app)
        for mode in ("user", "app", None):
            server = adapter._server_from_tool(tool(mode), workspace_client_for=resolver)
            assert server.workspace_client is (user if mode == "user" else app)
        assert [call.args for call in resolver.call_args_list] == [("user",), ("app",), ("app",)]
    adapter.workspace_client.assert_not_called()


def test_framework_adapters_share_auth_classification(adapter):
    from databricks_mason.runtime.mcp_auth import mcp_auth_error, mcp_tool_error

    assert adapter._auth_error is mcp_auth_error
    assert adapter._tool_error is mcp_tool_error


def test_explicit_deployed_user_without_resolver_fails_closed(adapter, monkeypatch):
    from databricks_mason.runtime.auth import AuthError

    monkeypatch.setenv("DATABRICKS_APP_NAME", "my-agent")
    with pytest.raises(AuthError) as raised:
        adapter._server_from_tool(tool("user"))
    assert raised.value.code == "MCP_USER_AUTHORIZATION_MISSING"
    assert raised.value.integration_id == "search"
    adapter.workspace_client.assert_not_called()


def test_legacy_and_local_default_identity_remain_supported(adapter):
    for mode in (None, "app", "user"):
        assert (
            adapter._server_from_tool(tool(mode)).workspace_client
            is adapter.workspace_client.return_value
        )


def test_extra_servers_are_not_resolved_or_modified(adapter):
    extra = FakeServer("customer", "https://customer.example/mcp", object())
    resolver = Mock(side_effect=AssertionError("must not resolve customer credentials"))
    build = getattr(adapter, "mcp_tools", None) or adapter.mcp_servers
    assert asyncio.run(build([extra], workspace_client_for=resolver)) == [extra]
    assert extra.url == "https://customer.example/mcp"
    resolver.assert_not_called()


def test_resolver_auth_errors_are_not_swallowed(adapter):
    from databricks_mason.runtime.auth import AuthError

    error = AuthError("MCP_PERMISSION_DENIED", "Denied", 403, "search")
    adapter.load_tools.return_value = [tool("user")]
    resolver = Mock(side_effect=error)
    build = getattr(adapter, "mcp_tools", None) or adapter.mcp_servers
    with pytest.raises(AuthError) as raised:
        asyncio.run(build(workspace_client_for=resolver))
    assert raised.value is error


def test_sandbox_reconnect_captures_resolver_and_manifest(adapter, monkeypatch):
    if not adapter.__name__.endswith("langgraph.mcp"):
        pytest.skip("LangGraph reconnect interceptor")
    user = SimpleNamespace(config=SimpleNamespace(host="https://workspace"))
    resolver = Mock(return_value=user)
    sandbox = tool("user", "sandbox", "sandbox")
    adapter.load_tools.return_value = [sandbox]
    clients = []

    class Client:
        def __init__(self, servers, **kwargs):
            self.interceptors = kwargs["tool_interceptors"]
            clients.append(self)

        async def get_tools(self, server_name=None):
            return []

    connections = []
    session = SimpleNamespace(initialize=AsyncMock(), call_tool=AsyncMock(return_value="ok"))

    @asynccontextmanager
    async def create_session(connection):
        connections.append(connection)
        yield session

    monkeypatch.setattr(adapter, "DatabricksMultiServerMCPClient", Client)
    monkeypatch.setattr(adapter, "create_session", create_session)
    asyncio.run(adapter.mcp_tools(workspace_client_for=resolver))
    adapter.load_tools.side_effect = AssertionError("manifest must not be reloaded mid-request")
    request = SimpleNamespace(server_name="sandbox", name="python", args={})
    for _ in range(2):
        assert asyncio.run(clients[0].interceptors[0](request, AsyncMock())) == "ok"
    assert [connection["workspace_client"] for connection in connections] == [user, user]
    adapter.workspace_client.assert_not_called()


@pytest.mark.parametrize("operation", ["connect", "list_tools", "call_tool"])
@pytest.mark.parametrize("auth", ["user", "app"])
def test_openai_only_user_permission_errors_are_typed(adapter, monkeypatch, operation, auth):
    if not adapter.__name__.endswith("openai.mcp"):
        pytest.skip("OpenAI connection lifecycle")
    from databricks_mason.runtime.auth import AuthError

    monkeypatch.setattr(
        FakeServer, operation, AsyncMock(side_effect=PermissionError("secret upstream body"))
    )
    server = adapter._server_from_tool(tool(auth))
    args = ("search", {}) if operation == "call_tool" else ()
    if auth == "app":
        with pytest.raises(PermissionError, match="secret upstream body"):
            asyncio.run(getattr(server, operation)(*args))
    else:
        with pytest.raises(AuthError) as raised:
            asyncio.run(getattr(server, operation)(*args))
        assert raised.value.code == "MCP_PERMISSION_DENIED"
        assert raised.value.integration_id == "search"
        assert "secret" not in str(raised.value)


@pytest.mark.parametrize("status", [401, 403])
def test_nested_sdk_auth_failure_is_sanitized(adapter, status):
    import httpx

    exception_group = getattr(builtins, "ExceptionGroup", None)
    if exception_group is None:
        pytest.skip("ExceptionGroup requires Python 3.11 or newer")
    response = httpx.Response(status, request=httpx.Request("GET", "https://workspace"))
    inner = httpx.HTTPStatusError("Bearer secret", request=response.request, response=response)
    outer = RuntimeError("SDK wrapper with secret")
    outer.__cause__ = inner
    error = adapter._auth_error(exception_group("wrapped", [outer]), "configured")
    assert error is not None
    assert error.status_code == status
    assert error.integration_id == "configured"
    assert "secret" not in str(error)


def test_sdk_permission_type_is_classified(adapter):
    from databricks.sdk.errors import PermissionDenied

    error = adapter._auth_error(PermissionDenied("secret"), "configured")
    assert error.code == "MCP_PERMISSION_DENIED"


def test_tool_result_permission_errors_are_not_model_results(adapter, monkeypatch):
    from databricks_mason.runtime.auth import AuthError

    result = SimpleNamespace(
        isError=True,
        structuredContent={"error": {"code": "PERMISSION_DENIED", "message": "secret"}},
    )
    if adapter.__name__.endswith("openai.mcp"):
        monkeypatch.setattr(FakeServer, "call_tool", AsyncMock(return_value=result))
        invoke = adapter._server_from_tool(tool("user")).call_tool("search", {})
    else:
        interceptor = adapter._sandbox_interceptor((tool("user"),))
        request = SimpleNamespace(server_name="search", name="search", args={})
        invoke = interceptor(request, AsyncMock(return_value=result))
    with pytest.raises(AuthError) as raised:
        asyncio.run(invoke)
    assert raised.value.code == "MCP_PERMISSION_DENIED"
    assert "secret" not in str(raised.value)


def test_app_tool_result_permission_errors_remain_model_results(adapter, monkeypatch):
    result = SimpleNamespace(
        isError=True,
        structuredContent={"error": {"code": "PERMISSION_DENIED", "message": "model-visible"}},
    )
    if adapter.__name__.endswith("openai.mcp"):
        monkeypatch.setattr(FakeServer, "call_tool", AsyncMock(return_value=result))
        invoke = adapter._server_from_tool(tool("app")).call_tool("search", {})
    else:
        interceptor = adapter._sandbox_interceptor((tool("app"),))
        request = SimpleNamespace(server_name="search", name="search", args={})
        invoke = interceptor(request, AsyncMock(return_value=result))

    assert asyncio.run(invoke) is result


def test_configured_discovery_permission_errors_are_not_hidden(adapter, monkeypatch):
    if not adapter.__name__.endswith("langgraph.mcp"):
        pytest.skip("LangGraph discovery")
    from databricks_mason.runtime.auth import AuthError

    adapter.load_tools.return_value = [tool("user")]
    monkeypatch.setattr(
        adapter.DatabricksMultiServerMCPClient,
        "get_tools",
        AsyncMock(side_effect=PermissionError("secret")),
    )
    with pytest.raises(AuthError) as raised:
        asyncio.run(adapter.mcp_tools())
    assert raised.value.code == "MCP_PERMISSION_DENIED"
    assert raised.value.integration_id == "search"


def test_openai_sdk_failure_handler_preserves_typed_auth_error(adapter):
    if not adapter.__name__.endswith("openai.mcp"):
        pytest.skip("OpenAI SDK error pipeline")
    from databricks_mason.runtime.auth import AuthError

    error = AuthError("MCP_PERMISSION_DENIED", "Denied", 403, "search")
    wrapper = RuntimeError("SDK wrapper")
    wrapper.__cause__ = error
    server = adapter._server_from_tool(tool("user"))
    with pytest.raises(AuthError) as raised:
        server.kwargs["failure_error_function"](None, wrapper)
    assert raised.value is error


def test_ordinary_run_code_error_roundtrips_to_model(adapter, monkeypatch):
    result = SimpleNamespace(
        isError=True,
        content=[{"type": "text", "text": "SyntaxError: invalid syntax"}],
        structuredContent={"error": {"code": "PYTHON_ERROR"}},
    )
    if adapter.__name__.endswith("openai.mcp"):
        monkeypatch.setattr(FakeServer, "call_tool", AsyncMock(return_value=result))
        invoke = adapter._server_from_tool(tool("app")).call_tool("run_code", {})
    else:
        interceptor = adapter._sandbox_interceptor((tool("app"),))
        request = SimpleNamespace(server_name="search", name="run_code", args={})
        invoke = interceptor(request, AsyncMock(return_value=result))
        client = adapter.mcp_client([adapter._server_from_tool(tool("app"))], tools=(tool("app"),))
        assert client.handle_tool_errors is True
    assert asyncio.run(invoke) is result
