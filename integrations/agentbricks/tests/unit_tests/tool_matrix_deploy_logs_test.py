"""Failure evidence from deployed App invocations."""

import pathlib
import runpy
import subprocess

_MATRIX = pathlib.Path(__file__).resolve().parents[1] / "e2e" / "tool_matrix.py"
_NAMESPACE = runpy.run_path(str(_MATRIX))
Runner = _NAMESPACE["Runner"]
ProjectCase = _NAMESPACE["ProjectCase"]
EvidenceRow = _NAMESPACE["EvidenceRow"]


def _stub_grant_snapshot(monkeypatch, runner) -> None:
    snapshot = {
        "tool_resources": [],
        "unrelated_resources": [],
        "uc_effective": {},
        "transitive_direct_privileges": [],
        "transitive_effective_privileges": [],
    }
    monkeypatch.setattr(runner, "_grant_snapshot", lambda app: snapshot)


def test_deploy_invocation_failure_captures_app_logs(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")
    runner = Runner(None, tmp_path, wheel)
    case = ProjectCase("langgraph", "cli", tmp_path, "test-app")
    deploy_log = tmp_path / "logs" / "deploy-langgraph-cli.log"
    commands = []

    monkeypatch.setattr(runner, "run_long", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "_assert_app_absent", lambda name: None)
    monkeypatch.setattr(runner, "_wait_for_app", lambda name: {"url": "https://test-app"})
    monkeypatch.setattr(runner, "_grant_transitive_function", lambda app: {})
    _stub_grant_snapshot(monkeypatch, runner)

    def fail_invocation(*args, **kwargs):
        runner.rows.append(
            EvidenceRow(
                framework="langgraph",
                authoring="cli",
                runtime="deploy",
                tool_kind="sandbox",
                status="fail",
                command="invoke",
                expected="marker",
                actual="",
                duration_seconds=0.1,
                artifact_paths=[str(deploy_log)],
                app_name=case.app_name,
                error="HTTP 500",
            )
        )

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "App traceback: model denied\n", "")

    monkeypatch.setattr(runner, "_exercise", fail_invocation)
    monkeypatch.setattr(runner, "run", fake_run)

    runner.deploy(case)

    app_log = tmp_path / "logs" / "deploy-runtime-langgraph-cli.log"
    assert app_log.read_text() == "App traceback: model denied\n"
    assert str(app_log) in runner.rows[0].artifact_paths
    assert commands == [["databricks", "apps", "logs", "test-app", "--tail-lines", "200"]]


def test_deploy_setup_failure_preserves_error_if_app_logs_unavailable(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")
    runner = Runner(None, tmp_path, wheel)
    case = ProjectCase("langgraph", "cli", tmp_path, "test-app")

    def fail_deploy(*args, **kwargs):
        raise RuntimeError("deployment failed")

    def fail_logs(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "app not found")

    monkeypatch.setattr(runner, "run_long", fail_deploy)
    monkeypatch.setattr(runner, "_assert_app_absent", lambda name: None)
    monkeypatch.setattr(runner, "run", fail_logs)

    runner.deploy(case)

    assert all(row.error == "deployment failed" for row in runner.rows)
    app_log = tmp_path / "logs" / "deploy-runtime-langgraph-cli.log"
    assert "app not found" in app_log.read_text()
    assert all(str(app_log) in row.artifact_paths for row in runner.rows)


def test_successful_deploy_does_not_fetch_app_logs(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")
    runner = Runner(None, tmp_path, wheel)
    case = ProjectCase("langgraph", "cli", tmp_path, "test-app")

    monkeypatch.setattr(runner, "run_long", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner, "_assert_app_absent", lambda name: None)
    monkeypatch.setattr(runner, "_wait_for_app", lambda name: {"url": "https://test-app"})
    monkeypatch.setattr(runner, "_grant_transitive_function", lambda app: {})
    _stub_grant_snapshot(monkeypatch, runner)
    monkeypatch.setattr(runner, "_exercise", lambda *args, **kwargs: None)
    log_fetches = []

    def unexpected_log_fetch(*args, **kwargs):
        log_fetches.append(args)

    monkeypatch.setattr(runner, "run", unexpected_log_fetch)

    runner.deploy(case)

    assert log_fetches == []
    assert not (tmp_path / "logs" / "deploy-runtime-langgraph-cli.log").exists()


def test_app_log_timeout_is_recorded(tmp_path: pathlib.Path, monkeypatch) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")
    runner = Runner(None, tmp_path, wheel)
    case = ProjectCase("langgraph", "cli", tmp_path, "test-app")

    def time_out(*args, **kwargs):
        raise TimeoutError("log service did not respond")

    monkeypatch.setattr(runner, "run", time_out)

    log_path = runner._capture_app_logs(case)

    assert log_path is not None
    assert "Could not retrieve App logs: log service did not respond" in log_path.read_text()
