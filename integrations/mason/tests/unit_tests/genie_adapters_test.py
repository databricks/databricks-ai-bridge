"""Real framework schemas and invocation over the shared Genie runtime."""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SPACE = "a" * 32
OTHER_SPACE = "e" * 32
CONVERSATION = "b" * 32
MESSAGE = "c" * 32
ATTACHMENT = "d" * 32
TEMPLATES = pathlib.Path(__file__).parents[2] / "src/databricks_mason/templates"


@pytest.fixture(params=["langgraph", "openai"])
def framework(request):
    pytest.importorskip("langchain_core.tools" if request.param == "langgraph" else "agents")
    return request.param


@pytest.fixture
def project(tmp_path, monkeypatch, framework):
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    def write(bindings=""):
        (tmp_path / "agent.toml").write_text(
            f'schema_version = 1\n[agent]\nframework = "{framework}"\nserver = "mason"\n{bindings}',
            encoding="utf-8",
        )

    write()
    return write


def _binding(name="sales", space_id=SPACE, auth=None):
    auth_line = f'auth = "{auth}"\n' if auth is not None else ""
    return (
        f'\n[[tools]]\nid = "{name}"\n{auth_line}'
        f'source = {{ kind = "genie_agent", space_id = "{space_id}" }}\n'
    )


def _tools(framework):
    return importlib.import_module(f"databricks_mason.{framework}").genie_tools()


def _schema(native_tool, framework):
    if framework == "langgraph":
        return native_tool.args_schema.model_json_schema()
    return native_tool.params_json_schema


async def _invoke(native_tool, arguments, framework):
    if framework == "langgraph":
        return await native_tool.ainvoke(arguments)
    from agents.tool_context import ToolContext

    payload = json.dumps(arguments)
    context = ToolContext(
        context=None,
        tool_name=native_tool.name,
        tool_call_id="test-call",
        tool_arguments=payload,
    )
    result = await native_tool.on_invoke_tool(context, payload)
    return json.loads(result) if isinstance(result, str) else result


def _response(**values):
    return SimpleNamespace(as_dict=lambda: values)


@pytest.fixture
def sdk(monkeypatch):
    from databricks.sdk import WorkspaceClient

    from databricks_mason.runtime import genie

    client = MagicMock(spec=WorkspaceClient)
    client.config = SimpleNamespace(host="https://example.databricks.com/")
    client.genie.start_conversation.return_value.response = SimpleNamespace(
        conversation_id=CONVERSATION, message_id=MESSAGE
    )
    client.genie.create_message.return_value.response = SimpleNamespace(message_id=MESSAGE)
    client.genie.get_message.return_value = _response(
        status="COMPLETED", attachments=[{"text": {"content": "42"}}]
    )
    client.genie.get_message_attachment_query_result.return_value = _response(
        statement_response={
            "status": {"state": "SUCCEEDED"},
            "manifest": {
                "schema": {"columns": [{"name": "count", "type_name": "LONG"}]},
                "total_row_count": 2,
            },
            "result": {"data_array": [["42"], [None]]},
        }
    )
    client_factory = MagicMock(return_value=client)
    monkeypatch.setattr(genie, "workspace_client", client_factory)
    return client, client_factory


def test_unconfigured_project_has_no_native_tools(framework, project, sdk):
    assert _tools(framework) == []
    sdk[1].assert_not_called()


def test_explicit_missing_project_is_not_silently_ignored(framework, tmp_path, monkeypatch):
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="MASON_PROJECT_ROOT"):
        _tools(framework)


def test_malformed_manifest_is_not_silently_ignored(framework, project, tmp_path):
    (tmp_path / "agent.toml").write_text("tools = [", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not read"):
        _tools(framework)


def test_native_tool_schemas_capture_space_and_hide_credentials(framework, project, sdk):
    project(_binding() + _binding("finance", OTHER_SPACE))
    native_tools = _tools(framework)
    assert [native_tool.name for native_tool in native_tools] == [
        "sales_ask",
        "sales_poll",
        "sales_query_result",
        "finance_ask",
        "finance_poll",
        "finance_query_result",
    ]
    expected_arguments = [
        {"question", "conversation_id"},
        {"conversation_id", "message_id"},
        {"conversation_id", "message_id", "attachment_id"},
    ]
    for native_tool, properties in zip(native_tools[:3], expected_arguments, strict=True):
        schema = _schema(native_tool, framework)
        assert set(schema["properties"]) == properties
        assert native_tool.description
        if framework == "openai":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == properties
    ask_schema = _schema(native_tools[0], framework)
    assert {"type": "null"} in ask_schema["properties"]["conversation_id"]["anyOf"]
    sdk[1].assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation_id", [None, CONVERSATION])
async def test_real_native_ask_invokes_fixed_space(framework, project, sdk, conversation_id):
    project(_binding() + _binding("finance", OTHER_SPACE))
    native_tools = _tools(framework)
    for native_tool, space_id in [(native_tools[0], SPACE), (native_tools[3], OTHER_SPACE)]:
        result = await _invoke(
            native_tool, {"question": "How many?", "conversation_id": conversation_id}, framework
        )
        assert result["space_id"] == space_id
        assert result["conversation_id"] == CONVERSATION
        assert result["message_id"] == MESSAGE
        assert result["attachments"] == [{"text": {"content": "42"}}]
        if conversation_id is None:
            sdk[0].genie.start_conversation.assert_called_with(space_id, "How many?")
        else:
            sdk[0].genie.create_message.assert_called_with(space_id, CONVERSATION, "How many?")


@pytest.mark.asyncio
async def test_native_genie_uses_binding_auth_with_request_resolver(framework, project, sdk):
    project(_binding(auth="user"))
    resolver = MagicMock(return_value=sdk[0])
    tools = importlib.import_module(f"databricks_mason.{framework}").genie_tools(
        workspace_client_for=resolver
    )

    result = await _invoke(tools[0], {"question": "How many?", "conversation_id": None}, framework)

    assert result["status"] == "COMPLETED"
    resolver.assert_called_with("user")
    sdk[1].assert_not_called()


@pytest.mark.asyncio
async def test_native_genie_without_request_resolver_keeps_app_default(framework, project, sdk):
    project(_binding())

    result = await _invoke(
        _tools(framework)[0], {"question": "How many?", "conversation_id": None}, framework
    )

    assert result["status"] == "COMPLETED"
    sdk[1].assert_called()


def test_deployed_user_genie_never_falls_back_to_app_identity(framework, project, sdk, monkeypatch):
    from databricks_mason.runtime.auth import AuthError

    monkeypatch.setenv("DATABRICKS_APP_NAME", "genie-obo-test")
    project(_binding(auth="user"))

    with pytest.raises(AuthError) as error:
        _tools(framework)

    assert error.value.code == "MCP_USER_AUTHORIZATION_MISSING"
    assert error.value.integration_id == "sales"
    sdk[1].assert_not_called()


@pytest.mark.asyncio
async def test_real_native_poll_resumes_without_asking_again(framework, project, sdk):
    project(_binding())
    result = await _invoke(
        _tools(framework)[1], {"conversation_id": CONVERSATION, "message_id": MESSAGE}, framework
    )
    assert result["status"] == "COMPLETED"
    assert result["space_id"] == SPACE
    sdk[0].genie.get_message.assert_called_once_with(SPACE, CONVERSATION, MESSAGE)
    sdk[0].genie.start_conversation.assert_not_called()
    sdk[0].genie.create_message.assert_not_called()


@pytest.mark.asyncio
async def test_real_native_query_result_preserves_structured_rows(framework, project, sdk):
    project(_binding())
    result = await _invoke(
        _tools(framework)[2],
        {"conversation_id": CONVERSATION, "message_id": MESSAGE, "attachment_id": ATTACHMENT},
        framework,
    )
    assert result["columns"] == [{"name": "count", "type_name": "LONG"}]
    assert result["rows"] == [["42"], [None]]
    assert result["truncated"] is False
    sdk[0].genie.get_message_attachment_query_result.assert_called_once_with(
        SPACE, CONVERSATION, MESSAGE, ATTACHMENT
    )


def test_native_tools_ignore_mcp_bindings(framework, project, sdk):
    project('\n[[tools]]\nid = "workspace_genie"\nsource = { kind = "genie_one" }\n')
    assert _tools(framework) == []
    sdk[1].assert_not_called()


@pytest.fixture
def mcp_adapter(framework):
    pytest.importorskip(
        "databricks_langchain" if framework == "langgraph" else "databricks_openai.agents"
    )
    return importlib.import_module(f"databricks_mason.{framework}.mcp")


def test_mcp_skips_native_bindings_without_credentials(mcp_adapter, project, monkeypatch):
    project(_binding())
    workspace = MagicMock(side_effect=AssertionError("Native records must not authenticate MCP"))
    monkeypatch.setattr(mcp_adapter, "workspace_client", workspace)
    assert mcp_adapter._declared_servers() == []
    workspace.assert_not_called()


def test_genie_one_uses_workspace_mcp_url_and_timeout(
    framework, mcp_adapter, project, sdk, monkeypatch
):
    project('\n[[tools]]\nid = "workspace_genie"\nsource = { kind = "genie_one" }\n')
    monkeypatch.setattr(mcp_adapter, "workspace_client", sdk[1])
    if framework == "langgraph":
        monkeypatch.setattr(
            mcp_adapter, "workspace_headers", lambda: {"x-databricks-workspace-id": "123"}
        )
    servers = mcp_adapter._declared_servers()
    assert len(servers) == 1
    server = servers[0]
    assert server.name == "workspace_genie"
    assert server.workspace_client is sdk[0]
    if framework == "langgraph":
        connection = server.to_connection_dict()
        assert connection["headers"]["x-databricks-workspace-id"] == "123"
        assert connection["timeout"] == timedelta(seconds=120)
    else:
        connection = server.params
        assert connection["timeout"] == 120.0
    assert connection["url"] == "https://example.databricks.com/api/2.0/mcp/genie"


@pytest.mark.parametrize("framework", ["openai"], indirect=True)
@pytest.mark.parametrize("workspace_id", ["123", None])
def test_openai_genie_one_forwards_workspace_routing_headers(
    framework, mcp_adapter, project, sdk, monkeypatch, workspace_id
):
    project('\n[[tools]]\nid = "workspace_genie"\nsource = { kind = "genie_one" }\n')
    sdk[0].config.host = "https://accounts.azuredatabricks.net/"
    monkeypatch.setattr(mcp_adapter, "workspace_client", sdk[1])
    if workspace_id is None:
        monkeypatch.delenv("DATABRICKS_WORKSPACE_ID", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", workspace_id)

    server = mcp_adapter._declared_servers()[0]

    assert server.params["url"] == "https://accounts.azuredatabricks.net/api/2.0/mcp/genie"
    assert (server.params.get("headers") or {}) == (
        {"X-Databricks-Org-Id": "123"} if workspace_id else {}
    )
    assert server.params["timeout"] == 120.0
    assert server.workspace_client is sdk[0]


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_genie_export_is_lazy_without_optional_dependencies(framework):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import sys
for dependency in ('agents', 'langchain_core', 'databricks_langchain', 'databricks_openai'):
    sys.modules[dependency] = None
module = importlib.import_module('databricks_mason.{framework}')
assert 'genie_tools' in module.__all__
assert 'databricks_mason.{framework}.genie' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_template_includes_native_genie_tools(framework, project, monkeypatch):
    pytest.importorskip("databricks_langchain" if framework == "langgraph" else "databricks_openai")
    if framework == "langgraph":
        pytest.importorskip("langchain.agents")
    project(_binding())
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "agent" or name.startswith("agent.")
    }
    for name in saved_modules:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(TEMPLATES / f"agent-{framework}"))
    try:
        module = importlib.import_module("agent.agent")
        monkeypatch.setattr(module, "memory_tools", lambda actor: [])
        if framework == "langgraph":
            monkeypatch.setattr(module, "mcp_tools", AsyncMock(return_value=[]))
            monkeypatch.setattr(module, "_RoutedChatDatabricks", MagicMock())
            monkeypatch.setattr(module, "workspace_client", MagicMock())
            monkeypatch.setattr(module, "checkpointer", lambda: None)
            monkeypatch.setattr(module, "create_agent", lambda **kwargs: SimpleNamespace(**kwargs))
            agent = await module.create_agent_graph("user")
        else:
            agent = module.create_agent("user")
        assert {"sales_ask", "sales_poll", "sales_query_result"} <= {
            native_tool.name for native_tool in agent.tools
        }
    finally:
        for name in list(sys.modules):
            if name == "agent" or name.startswith("agent."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.mark.asyncio
async def test_template_routes_request_client_to_native_genie(framework, project, monkeypatch):
    pytest.importorskip("databricks_langchain" if framework == "langgraph" else "databricks_openai")
    if framework == "langgraph":
        pytest.importorskip("langchain.agents")
    project(_binding(auth="user"))
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "agent" or name.startswith("agent.")
    }
    for name in saved_modules:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(TEMPLATES / f"agent-{framework}"))
    try:
        module = importlib.import_module("agent.agent")
        monkeypatch.setattr(module, "memory_tools", lambda actor: [])
        native = MagicMock(return_value=[])
        monkeypatch.setattr(module, "genie_tools", native)
        resolver = MagicMock()
        if framework == "langgraph":
            monkeypatch.setattr(module, "mcp_tools", AsyncMock(return_value=[]))
            monkeypatch.setattr(module, "_RoutedChatDatabricks", MagicMock())
            monkeypatch.setattr(module, "workspace_client", MagicMock())
            monkeypatch.setattr(module, "checkpointer", lambda: None)
            monkeypatch.setattr(module, "create_agent", lambda **kwargs: SimpleNamespace(**kwargs))
            await module.create_agent_graph("user", workspace_client_for=resolver)
        else:
            module.create_agent("user", workspace_client_for=resolver)
        native.assert_called_once_with(workspace_client_for=resolver)
    finally:
        for name in list(sys.modules):
            if name == "agent" or name.startswith("agent."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
