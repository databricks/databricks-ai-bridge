"""Failure-reporting checks for the gated live tool matrix."""

import pathlib
import runpy
import subprocess

_WRAPPER = pathlib.Path(__file__).resolve().parents[1] / "integration_tests" / "test_tool_matrix.py"
_runtime_log_tails = runpy.run_path(str(_WRAPPER))["_runtime_log_tails"]
_failed_row_summary = runpy.run_path(str(_WRAPPER))["_failed_row_summary"]
_safe_process_tail = runpy.run_path(str(_WRAPPER))["_safe_process_tail"]


def test_safe_process_tail_omits_html_and_long_lines(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "synthetic-test-secret")
    output = (
        "attempt failed: synthetic-test-secret\n"
        "<!DOCTYPE html><html>" + "x" * 3000 + "</html>\n" + "y" * 1000 + "\n15 passed, 1 failed\n"
    )

    tail = _safe_process_tail(output)

    assert "<redacted>" in tail
    assert "<HTML response omitted>" in tail
    assert "<oversized line omitted>" in tail
    assert "15 passed, 1 failed" in tail
    assert "synthetic-test-secret" not in tail
    assert "x" * 100 not in tail


def test_failed_row_summary_names_failing_cell_without_html_or_secret(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "synthetic-test-secret")
    (tmp_path / "evidence.json").write_text(
        '{"rows": ['
        '{"framework":"langgraph","authoring":"cli","runtime":"deploy",'
        '"tool_kind":"mcp","status":"fail","error":'
        '"HTTP 500: synthetic-test-secret <!DOCTYPE html><html>large response</html>"},'
        '{"framework":"langgraph","authoring":"direct","runtime":"deploy",'
        '"tool_kind":"mcp","status":"pass","error":null}'
        "]}",
        encoding="utf-8",
    )

    summary = _failed_row_summary(tmp_path)

    assert "langgraph/cli/deploy/mcp" in summary
    assert "HTTP 500" in summary
    assert "synthetic-test-secret" not in summary
    assert "<html>" not in summary
    assert "langgraph/direct/deploy/mcp" not in summary


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


def test_runtime_log_tails_keeps_mcp_diagnostic_before_long_traceback(
    tmp_path: pathlib.Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "deploy-runtime-langgraph-cli.log").write_text(
        "MCP tool web_search failed: McpError (HTTP status=500)\n"
        + "x" * 7000
        + "\nAuthError: The configured MCP tool failed.\n",
        encoding="utf-8",
    )

    tails = _runtime_log_tails(tmp_path)

    assert "MCP tool web_search failed: McpError (HTTP status=500)" in tails


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
