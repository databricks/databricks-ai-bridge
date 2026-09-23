"""Nightly workspace integration test: mason's deploy-and-invoke tool matrix.

Gated by ``RUN_MASON_INTEGRATION_TESTS=1`` so it never runs in normal pytest or PR CI (matching the
other suites in this repo). It drives the end-to-end matrix in ``tests/e2e/tool_matrix.py`` against a
live workspace: it scaffolds LangGraph agents, runs each under ``mason dev`` *and* deploys each to
Databricks Apps, and exercises the sandbox, web search, a local Python tool, and a temporary Unity
Catalog function -- then asserts every matrix cell passed. Auth comes from ambient Databricks
environment credentials (the CI service principal), so no ``--profile`` is passed.

Environment:
    RUN_MASON_INTEGRATION_TESTS   "1" to enable this suite
    DATABRICKS_HOST / _CLIENT_ID / _CLIENT_SECRET   service-principal auth for the CLIs and SDK
    MASON_INTEGRATION_UC_SCHEMA   two-part ``catalog.schema`` the SP can create a scratch function in
    MASON_INTEGRATION_PREPROVISIONED_APP_CATALOG_ACCESS   "1" when Apps already have USE CATALOG
    MASON_INTEGRATION_WAREHOUSE_ID   optional; a SQL warehouse to use instead of auto-discovering one
    MASON_WHEEL                   optional; a prebuilt databricks-mason wheel (else one is built here)
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MASON_INTEGRATION_TESTS") != "1",
    reason="RUN_MASON_INTEGRATION_TESTS is not set",
)

_MASON_PKG = pathlib.Path(__file__).resolve().parents[2]  # integrations/mason
_TOOL_MATRIX = _MASON_PKG / "tests" / "e2e" / "tool_matrix.py"
# A two-part catalog.schema the CI service principal can create a scratch UC function in.
_UC_SCHEMA = os.environ.get("MASON_INTEGRATION_UC_SCHEMA", "main.mason_agent_tools_e2e")
_MATRIX_TIMEOUT_SECONDS = 45 * 60
_CLEANUP_GRACE_SECONDS = 10 * 60


def _wheel(tmp_path: pathlib.Path) -> pathlib.Path:
    """The databricks-mason wheel under test: a prebuilt ``MASON_WHEEL`` if set, else built here."""
    prebuilt = os.environ.get("MASON_WHEEL")
    if prebuilt:
        return pathlib.Path(prebuilt)
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=_MASON_PKG,
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
    warehouse = os.environ.get("MASON_INTEGRATION_WAREHOUSE_ID")
    if warehouse:
        argv += ["--warehouse-id", warehouse]
    if os.environ.get("MASON_INTEGRATION_PREPROVISIONED_APP_CATALOG_ACCESS") == "1":
        argv.append("--preprovisioned-app-catalog-access")

    # No --profile: tool_matrix falls back to ambient env credentials (the CI service principal),
    # which as OAuth also authorize the deployed App's /api calls. This deploys real Apps and starts
    # a warehouse, so it is slow. Reserve ten minutes before the outer job timeout for the child to
    # unwind through its cleanup; a hard kill is only the fallback after that grace period.
    result, timed_out = _run_matrix(argv)
    if timed_out or result.returncode != 0:
        evidence = output / "evidence.json"
        detail = evidence.read_text() if evidence.is_file() else "(no evidence.json written)"
        outcome = (
            f"timed out after {_MATRIX_TIMEOUT_SECONDS}s; cleanup was requested"
            if timed_out
            else f"exited {result.returncode}"
        )
        pytest.fail(
            f"tool_matrix {outcome}\n"
            f"STDOUT (tail):\n{result.stdout[-4000:]}\n"
            f"STDERR (tail):\n{result.stderr[-4000:]}\n"
            f"evidence (head):\n{detail[:4000]}"
        )
