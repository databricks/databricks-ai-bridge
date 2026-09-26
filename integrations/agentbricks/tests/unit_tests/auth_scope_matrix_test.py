"""Offline contract tests for the deployed declarative-auth evidence matrix."""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import tomli


def _load_matrix_module() -> ModuleType:
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"
    spec = importlib.util.spec_from_file_location("auth_scope_matrix_e2e", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _complete_evidence() -> dict:
    source_sha = "a" * 40
    marker = f"AUTH_SCOPE_E2E_{source_sha[:12]}"
    rows = []
    for framework in ("langgraph", "openai"):
        for scope_source in ("explicit", "combined"):
            required = ["sql"] if scope_source == "explicit" else ["ai-gateway", "sql"]
            rows.append(
                {
                    "framework": framework,
                    "scope_source": scope_source,
                    "status": "pass",
                    "required_scopes": required,
                    "configured_scopes": required,
                    "effective_scopes": [*required, "iam.current-user:read"],
                    "user_sql_marker": "AGENTBRICKS_USER_SQL_OK",
                    "app_sql_denied": True,
                    "web_search_verified": scope_source == "combined",
                    "invocation_status": "completed",
                    "freshness_marker": marker,
                }
            )
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "wheel_sha256": "b" * 64,
        "rows": rows,
        "cleanup": {"apps_deleted": True, "sql_asset_deleted": True},
    }


def _run_evidence_verifier(
    tmp_path: pathlib.Path, document: dict
) -> subprocess.CompletedProcess[str]:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"
    return subprocess.run(
        [sys.executable, str(script), "--verify-evidence", str(evidence)],
        text=True,
        capture_output=True,
        timeout=30,
    )


class _StatementExecution:
    def __init__(self, error: Exception | None = None, data_array=None):
        self.error = error
        self.data_array = [["AGENTBRICKS_USER_SQL_OK"]] if data_array is None else data_array
        self.statements = []

    def execute_statement(self, **kwargs):
        self.statements.append(kwargs["statement"])
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            status=SimpleNamespace(state="SUCCEEDED"),
            result=SimpleNamespace(data_array=self.data_array),
        )


class _WorkspaceClient:
    def __init__(self, error: Exception | None = None, data_array=None):
        self.statement_execution = _StatementExecution(error, data_array)


def _generated_sql_tool(tmp_path: pathlib.Path):
    matrix = _load_matrix_module()
    runner = matrix.Runner.__new__(matrix.Runner)
    runner.catalog = "catalog"
    runner.schema = "schema"
    runner.table_name = "table"
    runner.warehouse_id = "warehouse"
    runner.freshness_marker = "AUTH_SCOPE_E2E_aaaaaaaaaaaa"
    runner.transcript = matrix.Transcript(tmp_path / "commands.log")
    project = tmp_path / "project"
    (project / "agent").mkdir(parents=True)
    runner._write_auth_scope_tool(project, "langgraph")
    generated = project / "agent" / "auth_scope_tools.py"
    spec = importlib.util.spec_from_file_location("generated_auth_scope_tools", generated)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return matrix, module


@pytest.mark.parametrize(
    ("framework", "package", "subdirectory"),
    [
        ("langgraph", "databricks-langchain", "integrations/langchain"),
        ("openai", "databricks-openai", "integrations/openai"),
    ],
)
def test_runtime_source_pins_include_framework_package(
    tmp_path: pathlib.Path, framework: str, package: str, subdirectory: str
):
    matrix = _load_matrix_module()
    runner = matrix.Runner.__new__(matrix.Runner)
    runner.source_repo = "https://github.com/example/databricks-ai-bridge.git"
    runner.source_ref = "a" * 40
    runner.transcript = matrix.Transcript(tmp_path / "commands.log")
    project = tmp_path / framework
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'matrix'\n", encoding="utf-8")

    runner._pin_runtime_source(project, framework)

    sources = tomli.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"][
        "sources"
    ]
    expected_common = {
        "git": runner.source_repo,
        "rev": runner.source_ref,
    }
    assert sources == {
        "databricks-agentbricks": {
            **expected_common,
            "subdirectory": "integrations/agentbricks",
        },
        package: {**expected_common, "subdirectory": subdirectory},
    }


def test_auth_scope_matrix_accepts_complete_redacted_evidence(tmp_path: pathlib.Path):
    result = _run_evidence_verifier(tmp_path, _complete_evidence())

    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed, 0 failed, 0 skipped" in result.stdout


def test_auth_scope_matrix_rejects_missing_negative_control(tmp_path: pathlib.Path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": "a" * 40,
                "wheel_sha256": "b" * 64,
                "rows": [],
                "cleanup": {"apps_deleted": True, "sql_asset_deleted": True},
            }
        ),
        encoding="utf-8",
    )
    script = pathlib.Path(__file__).parents[1] / "e2e" / "auth_scope_matrix.py"

    result = subprocess.run(
        [sys.executable, str(script), "--verify-evidence", str(evidence)],
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "4 skipped" in result.stdout


@pytest.mark.parametrize(
    "case",
    [
        "configured_scopes",
        "effective_scopes",
        "user_marker",
        "app_denial",
        "freshness",
        "duplicate_cell",
        "unexpected_cell",
        "source_sha",
        "wheel_sha",
        "cleanup",
    ],
)
def test_auth_scope_matrix_rejects_tampered_evidence(tmp_path: pathlib.Path, case: str):
    document = _complete_evidence()
    if case == "configured_scopes":
        document["rows"][0]["configured_scopes"] = []
    elif case == "effective_scopes":
        document["rows"][0]["effective_scopes"] = ["iam.current-user:read"]
    elif case == "user_marker":
        document["rows"][0]["user_sql_marker"] = ""
    elif case == "app_denial":
        document["rows"][0]["app_sql_denied"] = False
    elif case == "freshness":
        document["rows"][0]["freshness_marker"] = "AUTH_SCOPE_E2E_stale"
    elif case == "duplicate_cell":
        document["rows"].append(copy.deepcopy(document["rows"][0]))
    elif case == "unexpected_cell":
        unexpected = copy.deepcopy(document["rows"][0])
        unexpected["framework"] = "unexpected"
        document["rows"].append(unexpected)
    elif case == "source_sha":
        document["source_sha"] = "short"
    elif case == "wheel_sha":
        document["wheel_sha256"] = "short"
    elif case == "cleanup":
        document["cleanup"]["apps_deleted"] = False

    result = _run_evidence_verifier(tmp_path, document)

    assert result.returncode == 1, result.stdout + result.stderr


def test_generated_sql_tool_rejects_non_permission_app_failure(tmp_path: pathlib.Path):
    _, generated = _generated_sql_tool(tmp_path)
    clients = {
        "app": _WorkspaceClient(RuntimeError("network unavailable")),
        "user": _WorkspaceClient(),
    }
    tool = generated.auth_scope_tools(clients.get)[0]

    with pytest.raises(RuntimeError, match="network unavailable"):
        tool.invoke({})


def test_generated_sql_tool_accepts_identity_filtered_app_control(tmp_path: pathlib.Path):
    _, generated = _generated_sql_tool(tmp_path)
    clients = {
        "app": _WorkspaceClient(data_array=[]),
        "user": _WorkspaceClient(),
    }
    tool = generated.auth_scope_tools(clients.get)[0]

    result = tool.invoke({})

    assert "AGENTBRICKS_USER_SQL_OK" in result
    assert "AGENTBRICKS_APP_SQL_DENIED:empty-result" in result
    assert clients["app"].statement_execution.statements == [
        "SELECT marker FROM `catalog`.`schema`.`table` WHERE owner = current_user()"
    ]


def test_web_search_evidence_rejects_text_without_tool_execution():
    matrix = _load_matrix_module()
    response = {
        "status": "completed",
        "output": [
            {
                "role": "assistant",
                "content": "Search result: https://docs.databricks.com/security/auth/oauth.html",
            }
        ],
    }

    assert matrix._has_completed_search_tool_call(response) is False


def test_web_search_evidence_accepts_completed_search_tool_call():
    matrix = _load_matrix_module()
    response = {
        "status": "completed",
        "output": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "web_search", "args": {"query": "OAuth scopes"}}],
            },
            {
                "role": "tool",
                "name": "web_search",
                "content": "https://docs.databricks.com/security/auth/oauth.html",
            },
        ],
    }

    assert matrix._has_completed_search_tool_call(response) is True


def test_invocation_readiness_retries_502_and_503(tmp_path: pathlib.Path):
    matrix = _load_matrix_module()
    outcomes = iter(
        [
            matrix.InvocationHTTPError(502, "App Not Available"),
            matrix.InvocationHTTPError(503, "App starting"),
            ({"status": "completed"}, 200),
        ]
    )
    attempts = 0

    def invoke():
        nonlocal attempts
        attempts += 1
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    transcript = matrix.Transcript(tmp_path / "commands.log")

    response, status = matrix._invoke_with_readiness_retry(
        "invoke-sql-langgraph-explicit",
        invoke,
        transcript,
        timeout=1,
        retry_interval=0,
    )

    assert (response, status) == ({"status": "completed"}, 200)
    assert attempts == 3
    log = transcript.path.read_text(encoding="utf-8")
    assert "attempt 1 | HTTP 502 | retrying" in log
    assert "attempt 2 | HTTP 503 | retrying" in log


def test_invocation_readiness_does_not_retry_functional_4xx(tmp_path: pathlib.Path):
    matrix = _load_matrix_module()
    attempts = 0

    def invoke():
        nonlocal attempts
        attempts += 1
        raise matrix.InvocationHTTPError(400, "Invalid invocation")

    with pytest.raises(matrix.InvocationHTTPError, match="HTTP 400"):
        matrix._invoke_with_readiness_retry(
            "invoke-sql-langgraph-explicit",
            invoke,
            matrix.Transcript(tmp_path / "commands.log"),
            timeout=1,
            retry_interval=0,
        )

    assert attempts == 1
