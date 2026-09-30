"""Nightly workspace integration test: Agent Bricks deploy-and-invoke tool matrix.

Gated by ``RUN_AGENTBRICKS_INTEGRATION_TESTS=1`` so it never runs in normal pytest or PR CI (matching the
other suites in this repo). It drives the end-to-end matrix in ``tests/e2e/tool_matrix.py`` against a
live workspace: it scaffolds LangGraph agents, runs each under ``agentbricks dev`` *and* deploys each to
Databricks Apps, and exercises the sandbox, web search, a local Python tool, and a temporary Unity
Catalog function -- then asserts every matrix cell passed. Auth comes from ambient Databricks
environment credentials (the CI service principal), so no ``--profile`` is passed.

Environment:
    RUN_AGENTBRICKS_INTEGRATION_TESTS   "1" to enable this suite
    DATABRICKS_HOST / _CLIENT_ID / _CLIENT_SECRET   service-principal auth for the CLIs and SDK
    AGENTBRICKS_INTEGRATION_UC_SCHEMA   two-part ``catalog.schema`` for a scratch function
    AGENTBRICKS_INTEGRATION_BRIDGE_SHA   optional; pin generated App packages to this bridge commit
    AGENTBRICKS_INTEGRATION_PREPROVISIONED_APP_CATALOG_ACCESS   "1" when Apps have USE CATALOG
    AGENTBRICKS_INTEGRATION_WAREHOUSE_ID   optional; overrides warehouse discovery
    AGENTBRICKS_WHEEL   optional; a prebuilt databricks-agentbricks wheel (else built here)
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENTBRICKS_INTEGRATION_TESTS") != "1",
    reason="RUN_AGENTBRICKS_INTEGRATION_TESTS is not set",
)

_AGENTBRICKS_PKG = pathlib.Path(__file__).resolve().parents[2]
_TOOL_MATRIX = _AGENTBRICKS_PKG / "tests" / "e2e" / "tool_matrix.py"
# A two-part catalog.schema the CI service principal can create a scratch UC function in.
_UC_SCHEMA = os.environ.get("AGENTBRICKS_INTEGRATION_UC_SCHEMA", "main.agentbricks_agent_tools_e2e")
_MATRIX_TIMEOUT_SECONDS = 45 * 60
_CLEANUP_GRACE_SECONDS = 10 * 60


def _wheel(tmp_path: pathlib.Path) -> pathlib.Path:
    """The databricks-agentbricks wheel under test, built unless supplied by CI."""
    prebuilt = os.environ.get("AGENTBRICKS_WHEEL")
    if prebuilt:
        return pathlib.Path(prebuilt)
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=_AGENTBRICKS_PKG,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    wheels = sorted(dist.glob("*.whl"))
    assert wheels, "uv build produced no wheel"
    return wheels[-1]


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _run_matrix(argv: list[str]) -> tuple[subprocess.CompletedProcess[str], bool]:
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_MATRIX_TIMEOUT_SECONDS)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr), False
    except subprocess.TimeoutExpired:
        # SIGINT raises KeyboardInterrupt in tool_matrix.py, which unwinds through its finally block.
        # SIGTERM would terminate Python immediately and bypass its resource cleanup.
        _signal_process_group(process, signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr), True


def _runtime_log_tails(output: pathlib.Path) -> str:
    log_dir = output / "logs"
    app_logs = sorted(log_dir.glob("deploy-runtime-*.log"))
    paths = (
        app_logs
        + sorted(log_dir.glob("dev-*.log"))
        + [path for path in sorted(log_dir.glob("deploy-*.log")) if path not in app_logs]
    )
    if not paths:
        return "(no runtime logs written)"

    excerpts = []
    for path in paths[:6]:
        try:
            with path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                log_file.seek(max(0, log_file.tell() - 6000))
                tail = log_file.read().decode("utf-8", errors="replace")
            if path in app_logs:
                diagnostics = [
                    line[:500]
                    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if "MCP tool " in line and " failed:" in line
                ]
                if diagnostics:
                    tail = "MCP diagnostics:\n" + "\n".join(diagnostics[-5:]) + "\n" + tail
            tail = "\n".join(
                line if len(line) <= 500 else "<oversized line omitted>"
                for line in tail.splitlines()
            )
        except OSError as exc:
            tail = f"(could not read log: {exc})"
        for credential_name in ("DATABRICKS_CLIENT_SECRET", "DATABRICKS_TOKEN"):
            credential = os.environ.get(credential_name)
            if credential:
                tail = tail.replace(credential, "<redacted>")
        excerpts.append(f"{path.name} (tail):\n{tail}")
    if len(paths) > 6:
        excerpts.append(f"({len(paths) - 6} more logs omitted)")
    return "\n\n".join(excerpts)


def _failed_row_summary(output: pathlib.Path) -> str:
    evidence = output / "evidence.json"
    if not evidence.is_file():
        return f"No evidence written; inspect {output / 'commands.log'}"
    try:
        rows = json.loads(evidence.read_text(encoding="utf-8"))["rows"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Could not read failed rows from {evidence}: {type(exc).__name__}"
    failures = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "fail":
            continue
        label = "/".join(
            str(row.get(key, "?")) for key in ("framework", "authoring", "runtime", "tool_kind")
        )
        detail = str(row.get("error") or "No error recorded").splitlines()[0]
        html = re.search(r"<(?:!doctype|html)\b", detail, flags=re.IGNORECASE)
        if html:
            detail = detail[: html.start()] + "<HTML response omitted>"
        for credential_name in ("DATABRICKS_CLIENT_SECRET", "DATABRICKS_TOKEN"):
            credential = os.environ.get(credential_name)
            if credential:
                detail = detail.replace(credential, "<redacted>")
        failures.append(f"{label}: {detail[:500]}")
    return "\n".join(failures) if failures else "No failed rows recorded in evidence.json"


def _safe_process_tail(output: str) -> str:
    output = re.sub(
        r"<(?:!doctype html|html\b).*?</html>",
        "<HTML response omitted>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for credential_name in ("DATABRICKS_CLIENT_SECRET", "DATABRICKS_TOKEN"):
        credential = os.environ.get(credential_name)
        if credential:
            output = output.replace(credential, "<redacted>")
    lines = output.splitlines()[-40:]
    return "\n".join(line if len(line) <= 500 else "<oversized line omitted>" for line in lines)


def test_tool_matrix_deploy_and_invoke(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "matrix"
    argv = [
        sys.executable,
        str(_TOOL_MATRIX),
        "--wheel",
        str(_wheel(tmp_path)),
        "--output",
        str(output),
        "--uc-schema",
        _UC_SCHEMA,
    ]
    warehouse = os.environ.get("AGENTBRICKS_INTEGRATION_WAREHOUSE_ID")
    if warehouse:
        argv += ["--warehouse-id", warehouse]
    bridge_sha = os.environ.get("AGENTBRICKS_INTEGRATION_BRIDGE_SHA")
    if bridge_sha:
        argv += ["--bridge-sha", bridge_sha]
    if os.environ.get("AGENTBRICKS_INTEGRATION_PREPROVISIONED_APP_CATALOG_ACCESS") == "1":
        argv.append("--preprovisioned-app-catalog-access")
    # TEMP: retain Apps for one debugging run; revert after.
    argv.append("--keep-resources")

    # No --profile: tool_matrix falls back to ambient env credentials (the CI service principal),
    # which as OAuth also authorize the deployed App's /api calls. This deploys real Apps and starts
    # a warehouse, so it is slow. Reserve ten minutes before the outer job timeout for the child to
    # unwind through its cleanup; a hard kill is only the fallback after that grace period.
    result, timed_out = _run_matrix(argv)
    if timed_out or result.returncode != 0:
        outcome = (
            f"timed out after {_MATRIX_TIMEOUT_SECONDS}s; cleanup was requested"
            if timed_out
            else f"exited {result.returncode}"
        )
        pytest.fail(
            f"tool_matrix {outcome}\n"
            f"failed matrix rows:\n{_failed_row_summary(output)}\n"
            f"STDOUT (tail):\n{_safe_process_tail(result.stdout)}\n"
            f"STDERR (tail):\n{_safe_process_tail(result.stderr)}\n"
            f"runtime logs:\n{_runtime_log_tails(output)}"
        )
