"""Failure-reporting checks for the gated live tool matrix."""

import pathlib
import runpy
import subprocess

_WRAPPER = pathlib.Path(__file__).resolve().parents[1] / "integration_tests" / "test_tool_matrix.py"
_runtime_log_tails = runpy.run_path(str(_WRAPPER))["_runtime_log_tails"]


def test_runtime_log_tails_reports_missing_logs(tmp_path: pathlib.Path) -> None:
    assert _runtime_log_tails(tmp_path) == "(no runtime logs written)"


def test_runtime_log_tails_includes_dev_and_deploy_errors(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "dev-langgraph-cli.log").write_text(
        "old output\n" + "x" * 7000 + "\nInvocation execution failed: model denied\n",
        encoding="utf-8",
    )
    (logs / "deploy-langgraph-cli.log").write_text("App deploy failed\n", encoding="utf-8")

    tails = _runtime_log_tails(tmp_path)

    assert "dev-langgraph-cli.log (tail):" in tails
    assert "Invocation execution failed: model denied" in tails
    assert "deploy-langgraph-cli.log (tail):" in tails
    assert "App deploy failed" in tails
    assert "old output" not in tails


def test_runtime_log_tails_bounds_number_of_logs(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(7):
        (logs / f"dev-{index}.log").write_text(f"log {index}\n", encoding="utf-8")

    tails = _runtime_log_tails(tmp_path)

    assert "dev-5.log (tail):" in tails
    assert "dev-6.log (tail):" not in tails
    assert "(1 more logs omitted)" in tails


def test_runtime_log_tails_prioritizes_deployed_app_errors(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(7):
        (logs / f"dev-{index}.log").write_text(f"dev log {index}\n", encoding="utf-8")
    (logs / "deploy-runtime-langgraph-cli.log").write_text(
        "App invocation traceback\n", encoding="utf-8"
    )

    tails = _runtime_log_tails(tmp_path)

    assert "deploy-runtime-langgraph-cli.log (tail):" in tails
    assert "App invocation traceback" in tails


def test_runtime_log_tails_redacts_workspace_secret(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "synthetic-test-secret")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "dev-langgraph-cli.log").write_text(
        "credential=synthetic-test-secret\n", encoding="utf-8"
    )

    tails = _runtime_log_tails(tmp_path)

    assert "credential=<redacted>" in tails
    assert "synthetic-test-secret" not in tails


def test_wrapper_forwards_bridge_sha_to_matrix(tmp_path: pathlib.Path, monkeypatch) -> None:
    wrapper = runpy.run_path(str(_WRAPPER))["test_tool_matrix_deploy_and_invoke"]
    namespace = wrapper.__globals__
    monkeypatch.setenv("AGENTBRICKS_INTEGRATION_BRIDGE_SHA", "a" * 40)
    monkeypatch.setitem(namespace, "_wheel", lambda _tmp_path: tmp_path / "agentbricks.whl")
    observed = []

    def run_matrix(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], bool]:
        observed.extend(argv)
        return subprocess.CompletedProcess(argv, 0, "", ""), False

    monkeypatch.setitem(namespace, "_run_matrix", run_matrix)

    wrapper(tmp_path)

    assert observed[observed.index("--bridge-sha") + 1] == "a" * 40
