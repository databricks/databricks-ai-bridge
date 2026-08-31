"""Behavior tests for Mason's manifest-driven MCP and Python-tool runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

_MASON_ROOT = pathlib.Path(__file__).parents[2]
_RUNTIME_MODULE = _MASON_ROOT / "src" / "databricks_mason" / "python_runtime.py"


class _Schema:
    def __init__(self, value: object) -> None:
        self._value = value

    def model_json_schema(self) -> object:
        return self._value


class _BaseTool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        schema: object,
        handler,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = _Schema(schema)
        self._handler = handler
        self.func = handler

    def get_input_schema(self) -> _Schema:
        return self.args_schema

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self._handler(**arguments)


def _tool(function):
    parameters = inspect.signature(function).parameters
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in parameters},
        "required": list(parameters),
    }
    return _BaseTool(
        name=function.__name__,
        description=inspect.getdoc(function) or "",
        schema=schema,
        handler=function,
    )


def _record(tool_id: str, entrypoint: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        source=SimpleNamespace(kind="python", entrypoint=entrypoint),
    )


def _load_runtime(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    records: tuple[SimpleNamespace, ...] | None,
    modules: dict[str, str],
):
    for name in tuple(sys.modules):
        if (
            name == "agent"
            or name.startswith("agent.")
            or name == "langchain_core"
            or name.startswith("langchain_core.")
        ):
            sys.modules.pop(name)

    tools_package = tmp_path / "agent" / "tools"
    mason_package = tmp_path / "agent" / "mason"
    tools_package.mkdir(parents=True)
    mason_package.mkdir(parents=True)
    (tmp_path / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (tools_package / "__init__.py").write_text("", encoding="utf-8")
    (mason_package / "__init__.py").write_text("", encoding="utf-8")
    for module_name, source in modules.items():
        (tools_package / f"{module_name}.py").write_text(source, encoding="utf-8")

    langchain_core = types.ModuleType("langchain_core")
    langchain_core.__path__ = []
    langchain_tools = types.ModuleType("langchain_core.tools")
    langchain_tools.__dict__["BaseTool"] = _BaseTool
    langchain_tools.__dict__["tool"] = _tool
    langchain_core.__dict__["tools"] = langchain_tools
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", langchain_tools)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.import_module("agent.mason")

    spec = importlib.util.spec_from_file_location(
        "databricks_mason.python_runtime", _RUNTIME_MODULE
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runtime)
    spec.loader.exec_module(runtime)
    if records is not None:
        monkeypatch.setattr(runtime, "_python_records", lambda: records)
    return runtime


def test_python_records_loads_the_configured_project_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(tmp_path, monkeypatch, records=None, modules={})
    (tmp_path / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "langgraph"

[[tools]]
id = "lookup-ticket"
source = { kind = "python", entrypoint = "agent.tools.lookup:lookup_ticket" }
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(tmp_path))

    records = runtime._python_records()

    assert len(records) == 1
    assert records[0].id == "lookup-ticket"
    assert records[0].source.entrypoint == "agent.tools.lookup:lookup_ticket"


def test_python_tools_resolves_only_exact_manifest_entrypoints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("declared-tool", "agent.tools.declared:declared_tool"),),
        modules={
            "declared": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def declared_tool(ticket_id: str) -> str:\n"
                '    """Look up a declared ticket."""\n'
                "    return ticket_id\n"
            ),
            "undeclared": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def undeclared_tool(value: str) -> str:\n"
                '    """This decorated tool is not activated."""\n'
                "    return value\n"
            ),
        },
    )

    assert [tool.name for tool in runtime.python_tools()] == ["declared_tool"]
    assert "agent.tools.undeclared" not in sys.modules


def test_resolve_python_tools_rejects_missing_module(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("missing", "agent.tools.missing:missing"),),
        modules={},
    )

    with pytest.raises(RuntimeError, match="Could not import.*agent.tools.missing"):
        runtime.resolve_python_tools()


def test_resolve_python_tools_rejects_missing_symbol(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("missing", "agent.tools.present:missing"),),
        modules={"present": "VALUE = True\n"},
    )

    with pytest.raises(RuntimeError, match="has no attribute.*missing"):
        runtime.resolve_python_tools()


def test_resolve_python_tools_rejects_non_base_tool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("plain", "agent.tools.plain:plain"),),
        modules={"plain": "def plain() -> str:\n    return 'plain'\n"},
    )

    with pytest.raises(RuntimeError, match="BaseTool"):
        runtime.resolve_python_tools()


def test_resolve_python_tools_rejects_normalized_name_collision(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(
            _record("lookup-ticket", "agent.tools.first:lookup_ticket"),
            _record("lookup_ticket", "agent.tools.second:lookup_ticket"),
        ),
        modules={
            "first": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def lookup_ticket(value: str) -> str:\n"
                '    """First lookup."""\n'
                "    return value\n"
            ),
            "second": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def lookup_ticket(value: str) -> str:\n"
                '    """Second lookup."""\n'
                "    return value\n"
            ),
        },
    )

    with pytest.raises(RuntimeError, match="collision"):
        runtime.resolve_python_tools()


def test_describe_python_tool_rejects_empty_description(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("empty", "agent.tools.empty:empty"),),
        modules={
            "empty": (
                "from langchain_core.tools import BaseTool\n\n"
                "empty = BaseTool(name='empty', description='', schema={'type': 'object'}, "
                "handler=lambda: 'ok')\n"
            )
        },
    )

    with pytest.raises(RuntimeError, match="description"):
        runtime.describe_python_tool("empty")


def test_describe_python_tool_rejects_invalid_schema(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("bad-schema", "agent.tools.bad_schema:bad_schema"),),
        modules={
            "bad_schema": (
                "from langchain_core.tools import BaseTool\n\n"
                "bad_schema = BaseTool(name='bad_schema', description='Bad schema.', schema=[], "
                "handler=lambda: 'ok')\n"
            )
        },
    )

    with pytest.raises(RuntimeError, match="schema"):
        runtime.describe_python_tool("bad-schema")


def test_describe_python_tool_returns_canonical_contract_fingerprint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("lookup-ticket", "agent.tools.lookup:lookup_ticket"),),
        modules={
            "lookup": (
                "from langchain_core.tools import BaseTool\n\n"
                "def _implementation(ticket_id: str) -> dict:\n"
                "    return {'ticket_id': ticket_id}\n\n"
                "lookup_ticket = BaseTool(\n"
                "    name='lookup_ticket',\n"
                "    description='Look up a ticket.',\n"
                "    schema={\n"
                "        'required': ['ticket_id'],\n"
                "        'properties': {'ticket_id': {'type': 'string'}},\n"
                "        'type': 'object',\n"
                "    },\n"
                "    handler=_implementation,\n"
                ")\n"
            )
        },
    )
    contract = {
        "id": "lookup-ticket",
        "entrypoint": "agent.tools.lookup:lookup_ticket",
        "description": "Look up a ticket.",
        "input_schema": {
            "required": ["ticket_id"],
            "properties": {"ticket_id": {"type": "string"}},
            "type": "object",
        },
        "implementation": (
            "def _implementation(ticket_id: str) -> dict:\n    return {'ticket_id': ticket_id}\n"
        ),
    }
    canonical_contract = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert runtime.describe_python_tool("lookup-ticket") == {
        "id": "lookup-ticket",
        "entrypoint": "agent.tools.lookup:lookup_ticket",
        "description": "Look up a ticket.",
        "input_schema": {
            "required": ["ticket_id"],
            "properties": {"ticket_id": {"type": "string"}},
            "type": "object",
        },
        "fingerprint": hashlib.sha256(canonical_contract.encode()).hexdigest(),
    }


def test_invoke_python_tool_rejects_non_json_result(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("non-json", "agent.tools.non_json:non_json"),),
        modules={
            "non_json": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def non_json() -> object:\n"
                '    """Return a non-JSON value."""\n'
                "    return object()\n"
            )
        },
    )

    with pytest.raises(RuntimeError, match="JSON"):
        runtime.invoke_python_tool("non-json", {})


def test_python_tool_lookup_rejects_undeclared_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(tmp_path, monkeypatch, records=(), modules={})

    with pytest.raises(RuntimeError, match="not declared"):
        runtime.describe_python_tool("undeclared")
    with pytest.raises(RuntimeError, match="not declared"):
        runtime.invoke_python_tool("undeclared", {})


def test_invoke_python_tool_returns_json_result(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("lookup-ticket", "agent.tools.lookup:lookup_ticket"),),
        modules={
            "lookup": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def lookup_ticket(ticket_id: str) -> dict:\n"
                '    """Look up a ticket."""\n'
                "    return {'ticket_id': ticket_id}\n"
            )
        },
    )

    assert runtime.invoke_python_tool("lookup-ticket", {"ticket_id": "INC-123"}) == {
        "ticket_id": "INC-123"
    }


def test_runtime_cli_writes_control_json_to_result_path_without_using_stdout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("lookup-ticket", "agent.tools.lookup:lookup_ticket"),),
        modules={
            "lookup": (
                "from langchain_core.tools import tool\n\n"
                "print('import log')\n\n"
                "@tool\n"
                "def lookup_ticket(ticket_id: str) -> dict:\n"
                '    """Look up a ticket."""\n'
                "    print('tool log')\n"
                "    return {'ticket_id': ticket_id}\n"
            )
        },
    )
    capsys.readouterr()
    result_path = tmp_path / "control.json"

    exit_code = runtime.main(
        [
            "run",
            "lookup-ticket",
            "--input",
            '{"ticket_id":"INC-123"}',
            "--result-path",
            str(result_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "ok": True,
        "tool": "lookup-ticket",
        "result": {"ticket_id": "INC-123"},
    }
    captured = capsys.readouterr()
    assert captured.out == "import log\ntool log\n"
    assert captured.err == ""


def test_runtime_cli_writes_a_control_error_for_a_broken_entrypoint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(_record("broken", "agent.tools.missing:broken"),),
        modules={},
    )
    result_path = tmp_path / "control.json"

    exit_code = runtime.main(["check", "broken", "--result-path", str(result_path)])

    assert exit_code == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert "Could not import module 'agent.tools.missing'" in payload["error"]


def test_named_runtime_check_does_not_resolve_unrelated_broken_tools(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(
            _record("valid", "agent.tools.valid:valid"),
            _record("broken", "agent.tools.missing:broken"),
        ),
        modules={
            "valid": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def valid(value: str) -> str:\n"
                '    """Return the supplied value."""\n'
                "    return value\n"
            )
        },
    )
    result_path = tmp_path / "check.json"

    exit_code = runtime.main(["check", "valid", "--result-path", str(result_path)])

    assert exit_code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert [tool["id"] for tool in payload["tools"]] == ["valid"]


def test_named_runtime_run_does_not_resolve_unrelated_broken_tools(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = _load_runtime(
        tmp_path,
        monkeypatch,
        records=(
            _record("valid", "agent.tools.valid:valid"),
            _record("broken", "agent.tools.missing:broken"),
        ),
        modules={
            "valid": (
                "from langchain_core.tools import tool\n\n"
                "@tool\n"
                "def valid(value: str) -> dict:\n"
                '    """Return the supplied value."""\n'
                "    return {'value': value}\n"
            )
        },
    )
    result_path = tmp_path / "run.json"

    exit_code = runtime.main(
        [
            "run",
            "valid",
            "--input",
            '{"value":"ok"}',
            "--result-path",
            str(result_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["result"] == {"value": "ok"}


def _write_direct_manifest(project: pathlib.Path) -> None:
    (project / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "langgraph"

[[tools]]
id = "sandbox"
source = { kind = "sandbox", service = "system.ai.sandbox" }
policy = { downscope = [{ resource = "table:samples.nyctaxi.trips", permission = "read_only" }] }

[[tools]]
id = "web"
source = { kind = "mcp", service = "system.ai.web_search" }

[[tools]]
id = "lookup"
source = { kind = "uc_function", function = "main.tools.lookup" }
""",
        encoding="utf-8",
    )


def _mcp_project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "langgraph"
    project.mkdir(parents=True)
    _write_direct_manifest(project)
    return project


class _FakeWorkspaceClient:
    def __init__(self):
        self.config = SimpleNamespace(host="https://df1.example.com")


def _reload_mcp():
    # The runtime modules read env at call time, but re-import so patched sys.modules take effect.
    for name in ("databricks_mason.langgraph.mcp", "databricks_mason.runtime.tool_manifest"):
        sys.modules.pop(name, None)
    return importlib.import_module("databricks_mason.langgraph.mcp")


def test_langgraph_runtime_loads_direct_manifest_and_protects_sandbox_meta(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _mcp_project(tmp_path)

    class FakeDatabricksMCPServer:
        def __init__(self, name, url, workspace_client=None, **kwargs):
            self.name = name
            self.url = url
            self.workspace_client = workspace_client
            self.kwargs = kwargs

        @classmethod
        def from_uc_function(
            cls, catalog, schema, name, function_name=None, workspace_client=None, **kwargs
        ):
            client = workspace_client or _FakeWorkspaceClient()
            return cls(
                name=name,
                url=(
                    f"{client.config.host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
                ),
                workspace_client=client,
                **kwargs,
            )

        def to_connection_dict(self):
            return {"transport": "streamable_http", "url": self.url}

    class FakeMultiServerClient:
        last = None

        def __init__(self, servers, **kwargs):
            self.servers = servers
            self.kwargs = kwargs
            FakeMultiServerClient.last = self

        async def get_tools(self):
            return [server.name for server in self.servers]

    class FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return name, arguments, kwargs

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            return False

    databricks = types.ModuleType("databricks")
    databricks_sdk = types.ModuleType("databricks.sdk")
    databricks_sdk.__dict__["WorkspaceClient"] = _FakeWorkspaceClient
    databricks_langchain = types.ModuleType("databricks_langchain")
    databricks_langchain.__dict__["DatabricksMCPServer"] = FakeDatabricksMCPServer
    databricks_langchain.__dict__["DatabricksMultiServerMCPClient"] = FakeMultiServerClient
    adapters = types.ModuleType("langchain_mcp_adapters")
    sessions = types.ModuleType("langchain_mcp_adapters.sessions")
    sessions.__dict__["create_session"] = lambda connection: FakeSessionContext()
    monkeypatch.setitem(sys.modules, "databricks", databricks)
    monkeypatch.setitem(sys.modules, "databricks.sdk", databricks_sdk)
    monkeypatch.setitem(sys.modules, "databricks_langchain", databricks_langchain)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", adapters)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.sessions", sessions)
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(project))

    mcp = _reload_mcp()

    # _declared_servers() builds one server per manifest tool, with the right URLs.
    servers = mcp._declared_servers()
    assert [s.url for s in servers] == [
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.sandbox",
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.web_search",
        "https://df1.example.com/api/2.0/mcp/functions/main/tools/lookup",
    ]

    # mcp_tools() includes the manifest servers and fetches from a client that carries the sandbox
    # interceptor (the manifest declares a sandbox tool).
    assert asyncio.run(mcp.mcp_tools()) == ["sandbox", "web", "lookup"]
    client = FakeMultiServerClient.last
    assert client is not None
    assert len(client.kwargs["tool_interceptors"]) == 1

    # the interceptor downscopes sandbox calls with the fixed meta from the manifest policy.
    interceptor = client.kwargs["tool_interceptors"][0]
    request = SimpleNamespace(
        server_name="sandbox",
        name="sandbox",
        args={"code": 'print("ok")'},
    )

    async def unexpected_handler(request):
        raise AssertionError("sandbox calls must use the fixed-meta session")

    name, arguments, kwargs = asyncio.run(interceptor(request, unexpected_handler))
    assert name == "sandbox"
    assert arguments == {"code": 'print("ok")'}
    assert kwargs["meta"] == {
        "downscope": {"tables": [{"name": "samples.nyctaxi.trips", "permission": "read_only"}]}
    }


def test_manifest_reader_rejects_wrong_framework(tmp_path: pathlib.Path, monkeypatch):
    project = _mcp_project(tmp_path)
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(project))
    sys.modules.pop("databricks_mason.runtime.tool_manifest", None)
    manifest = importlib.import_module("databricks_mason.runtime.tool_manifest")

    with pytest.raises(RuntimeError, match="framework"):
        manifest.load_tools(expected_framework="langgraph")
