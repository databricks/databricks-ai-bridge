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


class _RuntimeCleanupClient:
    def __init__(self, app: str, commands: list):
        self.app = app
        self.commands = commands
        self.principal = "test-app-sp"
        self.store = {
            "name": f"runtime-stores/{app}",
            "owner": {"app": {"name": app, "service_principal_id": self.principal}},
            "storage_backend": {"lakebase": {"branch": "projects/test/branches/test-branch"}},
        }
        self.get_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.app_delete_error: Exception | None = None
        self.role_get_error: Exception | None = None
        self.api_client = type("ApiClient", (), {"do": self._do})()
        self.apps = type("Apps", (), {"get": self._get_app, "delete": self._delete_app})()
        self.postgres = type("Postgres", (), {"get_role": self._get_role})()

    def _do(self, method, path):
        assert path == f"/api/2.0/agents/runtime-stores/{self.app}"
        if method == "GET":
            if self.get_error:
                raise self.get_error
            return self.store
        assert method == "DELETE"
        self.commands.append(["sdk", method, path])
        if self.delete_error:
            raise self.delete_error
        return {}

    def _get_app(self, app):
        assert app == self.app
        return type("App", (), {"service_principal_client_id": self.principal})()

    def _delete_app(self, app):
        assert app == self.app
        self.commands.append(["sdk", "apps", "delete", app])
        if self.app_delete_error:
            raise self.app_delete_error

    def _get_role(self, role):
        if self.role_get_error:
            raise self.role_get_error
        return object()


def _cleanup_runner(tmp_path, monkeypatch, *, authoring="direct"):
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    app = f"agent-bricks-t-la-{authoring[:2]}-123456"
    runner.apps.append(app)
    runner.cases.append(
        ProjectCase(
            "langgraph",
            authoring,
            tmp_path,
            app,
            memory_store_name="memory-stores/test-memory-id" if authoring == "cli" else None,
            session_store_name="test-session-store" if authoring == "cli" else None,
        )
    )
    commands = []
    client = _RuntimeCleanupClient(app, commands)
    monkeypatch.setitem(Runner.cleanup.__globals__, "WorkspaceClient", lambda profile: client)
    role = "projects/test/branches/test-branch/roles/test-role"

    def fake_databricks(args, **kwargs):
        assert args == ["postgres", "list-roles", "projects/test/branches/test-branch"]
        return [
            {
                "name": role,
                "status": {"postgres_role": client.principal, "identity_type": "SERVICE_PRINCIPAL"},
            }
        ]

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner, "databricks", fake_databricks)
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "_wait_for_app_deleted", lambda name, **kwargs: None)
    monkeypatch.setattr(runner, "_delete_deployment_source", lambda name: None)
    return runner, client, commands, role


@pytest.mark.parametrize("authoring", ["cli", "direct"])
def test_cleanup_deletes_runtime_and_role_with_or_without_bound_stores(
    tmp_path: pathlib.Path, monkeypatch, authoring
) -> None:
    runner, client, commands, role = _cleanup_runner(tmp_path, monkeypatch, authoring=authoring)

    runner.cleanup()

    expected = []
    if authoring == "cli":
        expected = [
            [
                str(runner.agentbricks),
                "memory",
                "stores",
                "delete",
                "memory-stores/test-memory-id",
                "--yes",
            ],
            [
                str(runner.agentbricks),
                "sessions",
                "stores",
                "delete",
                "test-session-store",
                "--yes",
            ],
        ]
    expected += [
        ["sdk", "DELETE", f"/api/2.0/agents/runtime-stores/{client.app}"],
        ["sdk", "apps", "delete", client.app],
        ["databricks", "postgres", "delete-role", role],
    ]
    assert commands == expected
    assert {
        "resource": f"runtime-store:{client.app}",
        "status": "deleted",
    } in runner.cleanup_results
    assert {"resource": f"lakebase-role:{role}", "status": "deleted"} in runner.cleanup_results
    assert runner.cleanup_complete is True


def test_cleanup_keeps_role_when_bound_store_deletion_fails(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, _ = _cleanup_runner(tmp_path, monkeypatch, authoring="cli")

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 1 if "memory" in argv else 0, "", "")

    monkeypatch.setattr(runner, "run", fake_run)
    runner.cleanup()

    assert ["sdk", "apps", "delete", client.app] in commands
    assert not any("delete-role" in command for command in commands)
    assert runner.cleanup_complete is False


def test_cleanup_retains_app_identity_when_runtime_store_deletion_fails(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, _ = _cleanup_runner(tmp_path, monkeypatch)
    client.delete_error = RuntimeError("resource is in use")

    runner.cleanup()

    assert commands == [["sdk", "DELETE", f"/api/2.0/agents/runtime-stores/{client.app}"]]
    assert {
        "resource": f"runtime-store:{client.app}",
        "status": "failed",
        "detail": "resource is in use",
    } in runner.cleanup_results
    assert runner.cleanup_complete is False


def test_cleanup_treats_missing_runtime_store_as_deleted(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, _ = _cleanup_runner(tmp_path, monkeypatch)
    client.get_error = _NAMESPACE["NotFound"]("missing")
    client.delete_error = _NAMESPACE["NotFound"]("missing")

    runner.cleanup()

    assert commands == [["sdk", "apps", "delete", client.app]]
    assert {
        "resource": f"runtime-store:{client.app}",
        "status": "deleted",
    } in runner.cleanup_results
    assert runner.cleanup_complete is True


def test_cleanup_handles_runtime_store_removed_after_ownership_check(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, role = _cleanup_runner(tmp_path, monkeypatch)
    client.delete_error = _NAMESPACE["NotFound"]("missing")

    runner.cleanup()

    assert commands == [
        ["sdk", "DELETE", f"/api/2.0/agents/runtime-stores/{client.app}"],
        ["sdk", "apps", "delete", client.app],
        ["databricks", "postgres", "delete-role", role],
    ]
    assert runner.cleanup_complete is True


def test_cleanup_handles_app_already_absent(tmp_path: pathlib.Path, monkeypatch) -> None:
    runner, client, _, _ = _cleanup_runner(tmp_path, monkeypatch)
    client.get_error = _NAMESPACE["NotFound"]("missing store")
    client.app_delete_error = _NAMESPACE["NotFound"]("missing App")

    runner.cleanup()

    assert runner.cleanup_complete is True
    assert {
        "resource": f"runtime-store:{client.app}",
        "status": "deleted",
    } in runner.cleanup_results
    assert any(
        row["resource"] == f"app:{client.app}" and row["status"] == "deleted"
        for row in runner.cleanup_results
    )


def test_cleanup_handles_role_removed_after_ownership_check(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, role = _cleanup_runner(tmp_path, monkeypatch)
    client.role_get_error = _NAMESPACE["NotFound"]("missing role")

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 1, "", "role delete failed")

    monkeypatch.setattr(runner, "run", fake_run)
    runner.cleanup()

    assert runner.cleanup_complete is True
    assert {"resource": f"lakebase-role:{role}", "status": "deleted"} in runner.cleanup_results


@pytest.mark.parametrize("field", ["resource_name", "owner_name", "owner_principal", "branch"])
def test_cleanup_retains_app_and_store_when_runtime_ownership_is_not_verified(
    tmp_path: pathlib.Path, monkeypatch, field
) -> None:
    runner, client, commands, _ = _cleanup_runner(tmp_path, monkeypatch)
    if field == "resource_name":
        client.store["name"] = "runtime-stores/other-app"
    elif field == "owner_name":
        client.store["owner"]["app"]["name"] = "other-app"
    elif field == "owner_principal":
        client.store["owner"]["app"]["service_principal_id"] = "other-app-sp"
    else:
        client.store["storage_backend"]["lakebase"]["branch"] = "projects/other/branches/test/extra"

    runner.cleanup()

    assert commands == []
    assert runner.cleanup_complete is False
    assert runner.cleanup_results[0]["resource"] == f"runtime-store:{client.app}"
    assert runner.cleanup_results[0]["status"] == "failed"


def test_cleanup_does_not_treat_runtime_lookup_failure_as_absence(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    runner, client, commands, _ = _cleanup_runner(tmp_path, monkeypatch)
    client.get_error = _NAMESPACE["DatabricksError"]("permission denied")

    runner.cleanup()

    assert commands == []
    assert runner.cleanup_complete is False


@pytest.mark.parametrize("bad_role", ["other_branch", "other_identity_type", "duplicate"])
def test_cleanup_retains_app_when_role_ownership_is_ambiguous(
    tmp_path: pathlib.Path, monkeypatch, bad_role
) -> None:
    runner, client, commands, role = _cleanup_runner(tmp_path, monkeypatch)
    record = {
        "name": role if bad_role != "other_branch" else "projects/other/branches/test/roles/sp",
        "status": {
            "postgres_role": client.principal,
            "identity_type": "USER" if bad_role == "other_identity_type" else "SERVICE_PRINCIPAL",
        },
    }
    monkeypatch.setattr(
        runner, "databricks", lambda args: [record] * (2 if bad_role == "duplicate" else 1)
    )

    runner.cleanup()

    assert commands == []
    assert runner.cleanup_complete is False


class _FakeWorkspaceClient:
    def __init__(self, delete_error: Exception | None = None):
        self.deleted: list[tuple[str, bool]] = []
        self._delete_error = delete_error
        self.current_user = type(
            "CurrentUser", (), {"me": lambda _: type("User", (), {"user_name": "ci-sp"})()}
        )()
        self.workspace = type("Workspace", (), {"delete": self._delete})()

    def _delete(self, path: str, recursive: bool = False) -> None:
        self.deleted.append((path, recursive))
        if self._delete_error is not None:
            raise self._delete_error


def test_delete_deployment_source_removes_synced_app_folder(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    client = _FakeWorkspaceClient()
    monkeypatch.setitem(Runner.cleanup.__globals__, "WorkspaceClient", lambda profile: client)

    runner._delete_deployment_source("test-app")

    path = "/Workspace/Users/ci-sp/agentbricks_deployments/test-app"
    assert client.deleted == [(path, True)]
    assert runner.cleanup_results == [{"resource": f"workspace:{path}", "status": "deleted"}]


def test_delete_deployment_source_treats_missing_folder_as_deleted(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    client = _FakeWorkspaceClient(delete_error=_NAMESPACE["NotFound"]("missing"))
    monkeypatch.setitem(Runner.cleanup.__globals__, "WorkspaceClient", lambda profile: client)

    runner._delete_deployment_source("test-app")

    assert runner.cleanup_results[0]["status"] == "deleted"


def test_delete_deployment_source_records_failure(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")
    runner = Runner(None, tmp_path, wheel)
    client = _FakeWorkspaceClient(delete_error=RuntimeError("timed out"))
    monkeypatch.setitem(Runner.cleanup.__globals__, "WorkspaceClient", lambda profile: client)

    runner._delete_deployment_source("test-app")

    assert runner.cleanup_results == [
        {
            "resource": "workspace:/Workspace/Users/ci-sp/agentbricks_deployments/test-app",
            "status": "failed",
            "detail": "timed out",
        }
    ]
