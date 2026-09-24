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
    rows = [
        {
            "framework": framework,
            "authoring": authoring,
            "runtime": runtime,
            "tool_kind": tool,
            "status": "pass",
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
    package = repo / "integrations" / "agentbricks" / "src" / "databricks_agentbricks"
    (package / "cli").mkdir(parents=True)
    sources = {
        "databricks_agentbricks/tool_access.py": b"TOOL_ACCESS = True\n",
        "databricks_agentbricks/app_resources.py": b"APP_RESOURCES = True\n",
        "databricks_agentbricks/cli/deploy.py": b"DEPLOY = True\n",
    }
    for member, content in sources.items():
        relative = pathlib.Path(member).relative_to("databricks_agentbricks")
        (package / relative).write_bytes(content)
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
    runner.uc_function = "main.tools.marker"
    runner.uc_table = "main.tools.matrix_table"
    runner.uc_volume = "main.tools.matrix_volume"
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
    runner.uc_function = "main.tools.marker"
    runner.uc_table = "main.tools.matrix_table"
    runner.uc_volume = "main.tools.matrix_volume"
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
    assert scopes == ["table:main.tools.matrix_table", "volume:main.tools.matrix_volume"]


def test_direct_authoring_includes_temporary_volume_scope(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    runner = tool_matrix.Runner("profile", tmp_path / "out", tmp_path / "wheel.whl")
    runner.uc_function = "main.tools.marker"
    runner.uc_table = "main.tools.matrix_table"
    runner.uc_volume = "main.tools.matrix_volume"
    runner.genie_space_id = "0" * 32

    runner._author_direct(project, "langgraph")

    manifest = tool_matrix.tomli.loads((project / "agent.toml").read_text())
    sandbox = next(tool for tool in manifest["tools"] if tool["id"] == "sandbox")
    assert sandbox["policy"]["downscope"] == [
        {"resource": "table:main.tools.matrix_table", "permission": "read_only"},
        {"resource": "volume:main.tools.matrix_volume", "permission": "read_only"},
    ]
