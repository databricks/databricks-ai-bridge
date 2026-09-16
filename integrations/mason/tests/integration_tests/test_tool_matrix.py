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
    MASON_INTEGRATION_WAREHOUSE_ID   optional; a SQL warehouse to use instead of auto-discovering one
    MASON_WHEEL                   optional; a prebuilt databricks-mason wheel (else one is built here)
"""

from __future__ import annotations

import os
import pathlib
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

    # No --profile: tool_matrix falls back to ambient env credentials (the CI service principal),
    # which as OAuth also authorize the deployed App's /api calls. This deploys real Apps and starts
    # a warehouse, so it is slow; tool_matrix bounds each step internally and the CI job timeout is
    # the outer bound. Exit 0 means every matrix cell passed (tool_matrix verifies its own evidence).
    result = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        evidence = output / "evidence.json"
        detail = evidence.read_text() if evidence.is_file() else "(no evidence.json written)"
        pytest.fail(
            f"tool_matrix exited {result.returncode}\n"
            f"STDOUT (tail):\n{result.stdout[-4000:]}\n"
            f"STDERR (tail):\n{result.stderr[-4000:]}\n"
            f"evidence (head):\n{detail[:4000]}"
        )
