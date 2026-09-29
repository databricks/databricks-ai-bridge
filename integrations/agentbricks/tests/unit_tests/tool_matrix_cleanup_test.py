"""Cleanup only the resources the live matrix created."""

import pathlib
import runpy
import subprocess

import pytest

_MATRIX = pathlib.Path(__file__).resolve().parents[1] / "e2e" / "tool_matrix.py"
_NAMESPACE = runpy.run_path(str(_MATRIX))
Runner = _NAMESPACE["Runner"]
ProjectCase = _NAMESPACE["ProjectCase"]
MatrixError = _NAMESPACE["MatrixError"]


def test_app_preflight_refuses_to_reuse_existing_app(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    monkeypatch.setattr(
        runner,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "existing app", ""),
    )

    with pytest.raises(MatrixError, match="already exists"):
        runner._assert_app_absent("test-app")


def test_provision_stores_records_only_created_resource_names(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    (tmp_path / "agent.toml").write_text(
        '[memory_store]\nname = "test-memory"\n[session_store]\nname = "test-session"\n',
        encoding="utf-8",
    )
    case = ProjectCase("langgraph", "cli", tmp_path, "test-app")

    def fake_run(argv, **kwargs):
        output = (
            '{"name":"memory-stores/owned","display_name":"test-memory"}'
            if "memory" in argv
            else '{"session_store_name":"test-session"}'
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(runner, "run", fake_run)

    runner._provision_stores(case)

    assert case.memory_store_name == "memory-stores/owned"
    assert case.session_store_name == "test-session"


def test_cleanup_deletes_stores_runtime_app_and_matching_lakebase_role(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    app = "agent-bricks-t-la-cl-123456"
    runner.apps.append(app)
    runner.cases.append(
        ProjectCase(
            "langgraph",
            "cli",
            tmp_path,
            app,
            memory_store_name="memory-stores/test-memory-id",
            session_store_name="test-session-store",
        )
    )
    commands = []

    def fake_databricks(args, **kwargs):
        if args[:2] == ["apps", "get"]:
            return {"service_principal_client_id": "test-app-sp"}
        if args[:2] == ["api", "get"]:
            return {
                "name": f"runtime-stores/{app}",
                "owner": {"app": {"name": app, "service_principal_id": "test-app-sp"}},
                "storage_backend": {"lakebase": {"branch": "projects/test/branches/test-branch"}},
            }
        if args[:2] == ["postgres", "list-roles"]:
            return [
                {
                    "name": "projects/test/branches/test-branch/roles/test-role",
                    "status": {
                        "postgres_role": "test-app-sp",
                        "identity_type": "SERVICE_PRINCIPAL",
                    },
                }
            ]
        raise AssertionError(args)

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner, "databricks", fake_databricks)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.cleanup()

    assert commands == [
        [
            str(runner.agentbricks),
            "memory",
            "stores",
            "delete",
            "memory-stores/test-memory-id",
            "--yes",
        ],
        [str(runner.agentbricks), "sessions", "stores", "delete", "test-session-store", "--yes"],
        [str(runner.agentbricks), "deployments", "delete", app, "--yes"],
        [
            "databricks",
            "postgres",
            "delete-role",
            "projects/test/branches/test-branch/roles/test-role",
        ],
    ]


def test_cleanup_keeps_role_when_store_deletion_fails(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    app = "agent-bricks-t-la-cl-123456"
    runner.apps.append(app)
    runner.cases.append(
        ProjectCase("langgraph", "cli", tmp_path, app, memory_store_name="memory-stores/owned")
    )
    monkeypatch.setattr(
        runner, "_app_role_target", lambda name: "projects/test/branches/test/roles/sp"
    )
    commands = []

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 1 if "memory" in argv else 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)

    runner.cleanup()

    assert any("deployments" in command for command in commands)
    assert not any("delete-role" in command for command in commands)


def test_role_lookup_rejects_other_app_owner(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)

    def fake_databricks(args, **kwargs):
        if args[:2] == ["apps", "get"]:
            return {"service_principal_client_id": "test-app-sp"}
        return {
            "name": "runtime-stores/test-app",
            "owner": {"app": {"name": "other-app", "service_principal_id": "test-app-sp"}},
            "storage_backend": {"lakebase": {"branch": "projects/test/branches/test"}},
        }

    monkeypatch.setattr(runner, "databricks", fake_databricks)

    assert runner._app_role_target("test-app") is None
