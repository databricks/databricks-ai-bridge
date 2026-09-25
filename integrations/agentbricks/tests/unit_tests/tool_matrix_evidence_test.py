"""Regression tests for the standalone tool-matrix evidence contract."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import zipfile

import pytest
from databricks.sdk.errors import NotFound

_MATRIX_PATH = pathlib.Path(__file__).parents[1] / "e2e" / "tool_matrix.py"
_SPEC = importlib.util.spec_from_file_location("agentbricks_tool_matrix", _MATRIX_PATH)
assert _SPEC is not None and _SPEC.loader is not None
tool_matrix = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tool_matrix
_SPEC.loader.exec_module(tool_matrix)


def _evidence(*, cleanup_required: bool, cleanup_status: str = "deleted") -> dict:
    table_marker = "AGENTBRICKS_TABLE_0123456789abcdef"
    volume_marker = "AGENTBRICKS_VOLUME_fedcba9876543210"
    rows = [
        {
            "framework": framework,
            "authoring": authoring,
            "runtime": runtime,
            "tool_kind": tool,
            "status": "pass",
            "expected": {
                "sandbox_table": table_marker,
                "sandbox_volume": volume_marker,
            }.get(tool, "semantic result"),
            "actual": {
                "sandbox_table": json.dumps({"output": table_marker}),
                "sandbox_volume": json.dumps({"output": volume_marker}),
            }.get(tool, json.dumps({"output": "semantic result"})),
        }
        for framework in tool_matrix.FRAMEWORKS
        for authoring in tool_matrix.AUTHORING_PATHS
        for runtime in tool_matrix.RUNTIMES
        for tool in tool_matrix.TOOL_KINDS
    ]
    grant_checks = []
    for authoring in tool_matrix.AUTHORING_PATHS:
        repeated = authoring == "cli"
        grant_checks.append(
            {
                "authoring": authoring,
                "initial": {
                    "transitive_direct_privileges": [],
                    "transitive_effective_privileges": [],
                },
                "repeat": {} if repeated else None,
                "repeat_deploy_idempotent": True if repeated else None,
                "manual_transitive_grant_applied": True,
                "post_manual_transitive_grant": {
                    "direct_privileges": ["EXECUTE"],
                    "effective_privileges": ["EXECUTE"],
                },
            }
        )
    return {
        "schema_version": 1,
        "commit_sha": "a" * 40,
        "source_provenance": {
            "source_head_sha": "a" * 40,
            "source_dirty": False,
            "wheel_source_matches": True,
            "wheel_source_sha256": {"databricks_agentbricks/tool_access.py": "b" * 64},
        },
        "template_repo": "/tmp/databricks-ai-bridge",
        "template_ref": "feature",
        "workspace_host": "https://example.cloud.databricks.com",
        "started_at": "2026-09-24T00:00:00+00:00",
        "ended_at": "2026-09-24T01:00:00+00:00",
        "versions": {
            "agentbricks": "ab 0.2",
            "databricks": "Databricks CLI v1",
            "uv": "uv 0.8",
            "python": "Python 3.12",
        },
        "table_marker": table_marker,
        "volume_marker": volume_marker,
        "volume_file_path": (
            "/Volumes/supervisor_agent/mason_agent_tools_e2e/matrix_volume/marker.txt"
        ),
        "cleanup_required": cleanup_required,
        "cleanup_complete": cleanup_required and cleanup_status == "deleted",
        "cleanup": [
            {
                "resource": "app:test",
                "status": cleanup_status,
                **(
                    {"confirmed_absent_at": "2026-09-24T01:00:00+00:00"}
                    if cleanup_status == "deleted"
                    else {}
                ),
            }
        ],
        "grant_checks": grant_checks,
        "rows": rows,
    }


def test_verify_evidence_rejects_failed_required_cleanup(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(_evidence(cleanup_required=True, cleanup_status="failed")))

    assert tool_matrix.verify_evidence(path) == 1


def test_verify_evidence_allows_explicit_keep_resources(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    evidence["cleanup"] = []
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 0


def test_verify_evidence_requires_provenance(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    evidence.pop("commit_sha")
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 1


def test_verify_evidence_requires_nonempty_template_provenance(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    evidence["template_repo"] = None
    evidence["template_ref"] = None
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 1


def test_verify_evidence_ties_wheel_template_ref_to_wheel_sha(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    evidence["template_repo"] = "wheel://databricks-agentbricks"
    evidence["template_ref"] = "wrong"
    evidence["wheel_sha256"] = "a" * 64
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 1


def test_verify_evidence_requires_direct_manual_transitive_grant(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    evidence["grant_checks"][0]["post_manual_transitive_grant"]["direct_privileges"] = []
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 1


def test_verify_evidence_requires_exact_sandbox_markers(tmp_path):
    path = tmp_path / "evidence.json"
    evidence = _evidence(cleanup_required=False)
    sandbox_volume = next(row for row in evidence["rows"] if row["tool_kind"] == "sandbox_volume")
    sandbox_volume["actual"] = json.dumps({"output": "volume read succeeded"})
    path.write_text(json.dumps(evidence))

    assert tool_matrix.verify_evidence(path) == 1


def test_live_matrix_requires_commit_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool_matrix.py",
            "--wheel",
            str(tmp_path / "wheel.whl"),
            "--output",
            str(tmp_path / "output"),
            "--genie-space-id",
            "0" * 32,
        ],
    )

    with pytest.raises(SystemExit):
        tool_matrix.parse_args()


def test_live_matrix_requires_source_root(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool_matrix.py",
            "--wheel",
            str(tmp_path / "wheel.whl"),
            "--output",
            str(tmp_path / "output"),
            "--commit-sha",
            "a" * 40,
            "--genie-space-id",
            "0" * 32,
        ],
    )

    with pytest.raises(SystemExit):
        tool_matrix.parse_args()


def test_live_matrix_requires_full_hex_commit_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool_matrix.py",
            "--wheel",
            str(tmp_path / "wheel.whl"),
            "--output",
            str(tmp_path / "output"),
            "--source-root",
            str(tmp_path),
            "--commit-sha",
            "not-a-commit",
            "--genie-space-id",
            "0" * 32,
        ],
    )

    with pytest.raises(SystemExit):
        tool_matrix.parse_args()


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_checkout_and_wheel(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, pathlib.Path]:
    repo = tmp_path / "repo"
    source_root = repo / "integrations" / "agentbricks" / "src"
    sources = {
        "databricks_agentkit/_api_client.py": b"WORKSPACE_CLIENT = True\n",
        "databricks_agentbricks/tool_access.py": b"TOOL_ACCESS = True\n",
        "databricks_agentbricks/app_resources.py": b"APP_RESOURCES = True\n",
        "databricks_agentbricks/cli/deploy.py": b"DEPLOY = True\n",
    }
    for member, content in sources.items():
        path = source_root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _git(repo, "init")
    _git(repo, "config", "user.name", "Agent Bricks Test")
    _git(repo, "config", "user.email", "agentbricks-test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit_sha = _git(repo, "rev-parse", "HEAD")
    wheel = tmp_path / "agentbricks.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, content in sources.items():
            archive.writestr(member, content)
    return repo, commit_sha, wheel


def test_source_provenance_ties_wheel_modules_to_claimed_checkout(tmp_path):
    repo, commit_sha, wheel = _source_checkout_and_wheel(tmp_path)

    provenance = tool_matrix._source_provenance(repo, commit_sha, wheel)

    assert provenance["source_head_sha"] == commit_sha
    assert provenance["source_dirty"] is False
    assert provenance["wheel_source_matches"] is True
    assert set(provenance["wheel_source_sha256"]) == {
        "databricks_agentkit/_api_client.py",
        "databricks_agentbricks/tool_access.py",
        "databricks_agentbricks/app_resources.py",
        "databricks_agentbricks/cli/deploy.py",
    }


def test_source_provenance_rejects_unrelated_wheel(tmp_path):
    repo, commit_sha, wheel = _source_checkout_and_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        contents = {member: archive.read(member) for member in tool_matrix._WHEEL_SOURCE_FILES}
    contents["databricks_agentbricks/tool_access.py"] = b"WRONG = True\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, content in contents.items():
            archive.writestr(member, content)

    with pytest.raises(tool_matrix.MatrixError, match="does not match source checkout"):
        tool_matrix._source_provenance(repo, commit_sha, wheel)


def test_source_provenance_rejects_unrelated_claimed_commit(tmp_path):
    repo, _commit_sha, wheel = _source_checkout_and_wheel(tmp_path)

    with pytest.raises(tool_matrix.MatrixError, match="does not match source checkout HEAD"):
        tool_matrix._source_provenance(repo, "f" * 40, wheel)


def test_project_runtime_is_pinned_to_vendored_wheel(tmp_path):
    wheel = tmp_path / "dist" / "databricks_agentbricks-0.2.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"tested wheel")
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text('[project]\nname = "agent"\n\n[tool.uv]\ndefault-groups = []\n')
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)

    runner._pin_project_wheel(project)

    vendored = project / "agentbricks_e2e_wheels" / wheel.name
    assert vendored.read_bytes() == wheel.read_bytes()
    document = tool_matrix.tomli.loads(pyproject.read_text())
    assert document["tool"]["uv"]["sources"]["databricks-agentbricks"] == {
        "path": f"agentbricks_e2e_wheels/{wheel.name}"
    }


def test_invoke_with_retry_pins_compatible_ai_gateway_model(monkeypatch, tmp_path):
    bodies = []

    def fake_http_json(url, body, headers):
        bodies.append(body)
        return {"status": "completed"}

    monkeypatch.setattr(tool_matrix, "_http_json", fake_http_json)
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")

    response = runner._invoke_with_retry("sandbox", "https://app", "prompt", {})

    assert response == {"status": "completed"}
    assert bodies[0]["input"]["model"] == "system.ai.gpt-5-2"


def test_curl_evidence_records_pinned_ai_gateway_model():
    command = tool_matrix._curl_command("https://app/api/invocations", "prompt", False)

    assert "system.ai.gpt-5-2" in command


def test_wait_for_app_deleted_polls_until_not_found(monkeypatch, tmp_path):
    calls = 0

    class Apps:
        def get(self, name):
            nonlocal calls
            calls += 1
            if calls == 1:
                return object()
            raise NotFound("gone")

    client = type("Client", (), {"apps": Apps()})()
    monkeypatch.setattr(tool_matrix, "WorkspaceClient", lambda profile: client)
    monkeypatch.setattr(tool_matrix.time, "sleep", lambda seconds: None)
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")

    runner._wait_for_app_deleted("agent-bricks-test", timeout=1)

    assert calls == 2


def test_cleanup_records_failed_when_app_absence_is_not_confirmed(tmp_path):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)
    runner.apps = ["agent-bricks-test"]
    runner.run = lambda *args, **kwargs: type(
        "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
    )()

    def fail_confirmation(name):
        raise tool_matrix.MatrixError("App still exists")

    runner._wait_for_app_deleted = fail_confirmation

    runner.cleanup()

    assert runner.cleanup_complete is False
    assert runner.cleanup_results == [
        {
            "resource": "app:agent-bricks-test",
            "status": "failed",
            "detail": "App still exists",
        }
    ]


def test_cli_authoring_records_app_auth_for_managed_deployed_tools(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manifest = project / "agent.toml"
    manifest.write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
    )
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")
    runner.uc_function = "supervisor_agent.mason_agent_tools_e2e.marker"
    runner.uc_table = "supervisor_agent.mason_agent_tools_e2e.matrix_table"
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.genie_space_id = "0" * 32
    bindings = []

    def fake_run(argv, **kwargs):
        if any("missing_service" in str(argument) for argument in argv) and "add" in argv:
            return type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": '{"error":{"code":"NOT_FOUND"}}',
                },
            )()
        if "add" in argv:
            kind = argv[argv.index("add") + 1]
            auth = argv[argv.index("--auth") + 1] if "--auth" in argv else "user"
            tool_id = {
                "sandbox": "sandbox",
                "mcp": "web_search",
                "uc-function": "agentbricks_uc_marker",
                "genie-agent": "genie",
            }[kind]
            bindings.append((tool_id, auth))
            manifest.write_text(
                'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
                + "".join(
                    f'\n[[tools]]\nid = "{binding_id}"\nauth = "{binding_auth}"\n'
                    for binding_id, binding_auth in bindings
                )
            )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    runner.run = fake_run

    runner._author_cli(project)

    tools = {tool["id"]: tool for tool in tool_matrix.tomli.loads(manifest.read_text())["tools"]}
    assert {tools[tool_id]["auth"] for tool_id in ("sandbox", "web_search", "genie")} == {"app"}
    sandbox_command = next(args for args in bindings if args[0] == "sandbox")
    assert sandbox_command == ("sandbox", "app")


def test_cli_authoring_includes_temporary_volume_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
    )
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")
    runner.uc_function = "supervisor_agent.mason_agent_tools_e2e.marker"
    runner.uc_table = "supervisor_agent.mason_agent_tools_e2e.matrix_table"
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.genie_space_id = "0" * 32
    commands: list[list[str]] = []

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        if any("missing_service" in str(argument) for argument in argv) and "add" in argv:
            return type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": '{"error":{"code":"NOT_FOUND"}}',
                },
            )()
        if "add" in argv:
            kind = argv[argv.index("add") + 1]
            tool_id = {
                "sandbox": "sandbox",
                "mcp": "web_search",
                "uc-function": "agentbricks_uc_marker",
                "genie-agent": "genie",
            }[kind]
            manifest = tool_matrix.tomli.loads((project / "agent.toml").read_text())
            existing = manifest.get("tools", [])
            existing.append({"id": tool_id, "auth": "app"})
            (project / "agent.toml").write_text(
                'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
                + "".join(
                    f'\n[[tools]]\nid = "{tool["id"]}"\nauth = "{tool["auth"]}"\n'
                    for tool in existing
                )
            )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    runner.run = fake_run
    runner._author_cli(project)

    sandbox = next(command for command in commands if "sandbox" in command and "add" in command)
    scopes = [sandbox[index + 1] for index, value in enumerate(sandbox) if value == "--scope"]
    assert scopes == [
        "table:supervisor_agent.mason_agent_tools_e2e.matrix_table",
        "volume:supervisor_agent.mason_agent_tools_e2e.matrix_volume",
    ]


def test_direct_authoring_includes_temporary_volume_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")
    runner.uc_function = "supervisor_agent.mason_agent_tools_e2e.marker"
    runner.uc_table = "supervisor_agent.mason_agent_tools_e2e.matrix_table"
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.genie_space_id = "0" * 32

    runner._author_direct(project, "langgraph")

    manifest = tool_matrix.tomli.loads((project / "agent.toml").read_text())
    sandbox = next(tool for tool in manifest["tools"] if tool["id"] == "sandbox")
    assert sandbox["policy"]["downscope"] == [
        {
            "resource": "table:supervisor_agent.mason_agent_tools_e2e.matrix_table",
            "permission": "read_only",
        },
        {
            "resource": "volume:supervisor_agent.mason_agent_tools_e2e.matrix_volume",
            "permission": "read_only",
        },
    ]


def test_create_uc_function_seeds_hidden_table_and_volume_markers(monkeypatch, tmp_path):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)
    statements: list[str] = []
    uploads: list[tuple[str, bytes, bool | None]] = []

    runner.sql = lambda statement: statements.append(statement) or {}

    class Files:
        def upload(self, path, contents, *, overwrite=None):
            uploads.append((path, contents.read(), overwrite))

    client = type("Client", (), {"files": Files()})()
    monkeypatch.setattr(tool_matrix, "WorkspaceClient", lambda profile: client)

    runner.create_uc_function("supervisor_agent.mason_agent_tools_e2e")

    assert runner.table_marker
    assert runner.volume_marker
    assert runner.table_marker != runner.volume_marker
    table_statement = next(
        statement for statement in statements if statement.startswith("CREATE TABLE")
    )
    assert f"SELECT '{runner.table_marker}' AS marker" in table_statement
    assert uploads == [
        (
            runner.volume_file_path,
            runner.volume_marker.encode(),
            True,
        )
    ]
    assert runner.volume_file_path == (
        f"/Volumes/supervisor_agent/mason_agent_tools_e2e/"
        f"{runner.uc_volume.rsplit('.', 1)[-1]}/marker.txt"
    )


def test_exercise_reads_table_and_volume_without_disclosing_markers(tmp_path):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)
    runner.uc_function = "supervisor_agent.mason_agent_tools_e2e.marker"
    runner.uc_table = "supervisor_agent.mason_agent_tools_e2e.matrix_table"
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.table_marker = "AGENTBRICKS_TABLE_0123456789abcdef"
    runner.volume_marker = "AGENTBRICKS_VOLUME_fedcba9876543210"
    runner.volume_file_path = (
        "/Volumes/supervisor_agent/mason_agent_tools_e2e/matrix_volume/marker.txt"
    )
    prompts: dict[str, str] = {}

    responses = {
        "sandbox_table": {
            "output": "AnalysisException: initial incorrect query\n" * 200 + runner.table_marker
        },
        "sandbox_volume": {"output": runner.volume_marker},
        "mcp": {
            "output": "web_search returned Databricks Model Context Protocol documentation "
            "at https://docs.databricks.com/"
        },
        "python": {"output": "AGENTBRICKS_PYTHON_OK"},
        "uc_function": {"output": "AGENTBRICKS_UC_OK:matrix"},
        "genie": {
            "output": "genie_ask returned a conversation_id and a sufficiently detailed "
            "description of the configured data source"
        },
    }

    def invoke(label, url, prompt, headers):
        tool_kind = label.rsplit("-", 1)[-1]
        prompts[tool_kind] = prompt
        return responses[tool_kind]

    runner._invoke_with_retry = invoke
    runner._write_evidence = lambda: None
    case = tool_matrix.ProjectCase("langgraph", "cli", tmp_path, "agent-bricks-test")

    runner._exercise(case, "dev", "http://localhost:8400", {}, tmp_path / "dev.log")

    sandbox_rows = {
        row.tool_kind: row for row in runner.rows if row.tool_kind.startswith("sandbox_")
    }
    assert set(sandbox_rows) == {"sandbox_table", "sandbox_volume"}
    assert {row.status for row in sandbox_rows.values()} == {"pass"}
    assert runner.table_marker in sandbox_rows["sandbox_table"].actual
    assert f"SELECT marker FROM {runner.uc_table}" in prompts["sandbox_table"]
    assert runner.volume_file_path in prompts["sandbox_volume"]
    assert all(runner.table_marker not in prompt for prompt in prompts.values())
    assert all(runner.volume_marker not in prompt for prompt in prompts.values())


def test_exercise_rejects_sandbox_response_without_exact_hidden_marker(tmp_path):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)
    runner.uc_function = "supervisor_agent.mason_agent_tools_e2e.marker"
    runner.uc_table = "supervisor_agent.mason_agent_tools_e2e.matrix_table"
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.table_marker = "AGENTBRICKS_TABLE_0123456789abcdef"
    runner.volume_marker = "AGENTBRICKS_VOLUME_fedcba9876543210"
    runner.volume_file_path = (
        "/Volumes/supervisor_agent/mason_agent_tools_e2e/matrix_volume/marker.txt"
    )

    def invoke(label, url, prompt, headers):
        tool_kind = label.rsplit("-", 1)[-1]
        if tool_kind == "sandbox_table":
            return {"output": runner.table_marker}
        if tool_kind == "sandbox_volume":
            return {"output": "volume read succeeded"}
        if tool_kind == "mcp":
            return {
                "output": "web_search returned a Databricks result at "
                "https://docs.databricks.com/ with enough response detail"
            }
        if tool_kind == "python":
            return {"output": "AGENTBRICKS_PYTHON_OK"}
        if tool_kind == "uc_function":
            return {"output": "AGENTBRICKS_UC_OK:matrix"}
        return {
            "output": "genie_ask returned a conversation_id and a sufficiently detailed "
            "description of the configured data source"
        }

    runner._invoke_with_retry = invoke
    runner._write_evidence = lambda: None
    case = tool_matrix.ProjectCase("langgraph", "cli", tmp_path, "agent-bricks-test")

    runner._exercise(case, "dev", "http://localhost:8400", {}, tmp_path / "dev.log")

    rows = {row.tool_kind: row for row in runner.rows}
    assert rows["sandbox_table"].status == "pass"
    assert rows["sandbox_volume"].status == "fail"
    assert runner.volume_marker in rows["sandbox_volume"].error


def test_cleanup_deletes_volume_marker_file_before_dropping_volume(monkeypatch, tmp_path):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"wheel")
    runner = tool_matrix.Runner("profile", tmp_path / "out", wheel)
    runner.uc_volume = "supervisor_agent.mason_agent_tools_e2e.matrix_volume"
    runner.volume_file_path = (
        "/Volumes/supervisor_agent/mason_agent_tools_e2e/matrix_volume/marker.txt"
    )
    events: list[tuple[str, str]] = []

    class Files:
        def delete(self, path):
            events.append(("file", path))

    client = type("Client", (), {"files": Files()})()
    monkeypatch.setattr(tool_matrix, "WorkspaceClient", lambda profile: client)
    runner.sql = lambda statement: events.append(("sql", statement)) or {}
    runner._write_evidence = lambda: None

    runner.cleanup()

    assert events == [
        ("file", runner.volume_file_path),
        (
            "sql",
            "DROP VOLUME IF EXISTS `supervisor_agent`.`mason_agent_tools_e2e`.`matrix_volume`",
        ),
    ]
    assert runner.cleanup_results == [
        {"resource": f"file:{runner.volume_file_path}", "status": "deleted"},
        {"resource": f"volume:{runner.uc_volume}", "status": "deleted"},
    ]
    assert runner.cleanup_complete is True
