"""Functional e2e for `ab tracing`'s local dev store, read back by the REAL CLI.

Writes a trace to the sqlite-backed MLflow server that `ab dev` starts, then reads it with the
actual `ab tracing list` / `get`, which spin up a short-lived server and read over REST (never
opening the sqlite file directly). This is the loop we otherwise verify by hand. Fully local: an
unbound project resolves straight to the local store, so no workspace or auth is used.

Gated on `uv`/`uvx` and the `agentbricks` CLI on PATH; skipped otherwise (and skipped when the local MLflow
server can't be provisioned in the sandbox). Excluded from the default unit run (see `testpaths`), so
it runs opt-in / in CI, not on every `pytest`.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

# No [tracing] table -> tracing is unbound, so list/get resolve straight to the local dev store
# without any workspace lookup, keeping this test hermetic.
_AGENT_TOML = 'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "agentbricks"\n'


@pytest.fixture
def local_trace_store(tmp_path: pathlib.Path) -> pathlib.Path:
    """A project dir whose ``.agentbricks/mlflow.db`` holds one real trace.

    Uses agentbricks's own ``start_local_tracing_server`` (the real ``uvx mlflow`` server `ab dev` runs),
    logs a trace over REST with the mlflow client, and tears the server down (guaranteed, so nothing is
    orphaned). Skips when uv/uvx or the local server aren't available in the sandbox.
    """
    if shutil.which("uv") is None or shutil.which("uvx") is None:
        pytest.skip("requires uv/uvx on PATH")
    import mlflow

    from databricks_agentbricks.cli import tracing as tracing_mod

    project = tmp_path / "agent"
    project.mkdir()
    (project / "agent.toml").write_text(_AGENT_TOML)

    server, env = tracing_mod.start_local_tracing_server(project)
    if server is None:
        pytest.skip("local MLflow server could not be started")
    try:
        uri = env["MLFLOW_TRACKING_URI"]
        if not tracing_mod._wait_for_server(uri, server, timeout=180):
            pytest.skip("local MLflow server did not become ready")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(env["MLFLOW_EXPERIMENT_NAME"])
        with mlflow.start_span(name="probe") as span:
            span.set_inputs({"q": "hi"})
            span.set_outputs({"a": "there"})
        try:
            mlflow.flush_trace_async_logging(terminate=True)
        except Exception:  # noqa: BLE001 - older mlflow: let the export settle instead
            time.sleep(3)
    finally:
        tracing_mod.stop_local_tracing_server(server)
    return project


def _ab_json(*args: str, project: pathlib.Path, home: pathlib.Path):
    """Run the real `ab` CLI with `--output json`, isolated HOME, and return the parsed stdout.

    Inherits PATH so the read command can find `uvx`, but points HOME at an empty dir (no ab login /
    real ~/.agentbricks) and silences MLflow's agent hint so stdout is clean JSON.
    """
    ab = pathlib.Path(sys.executable).with_name("ab")
    if not ab.is_file():
        pytest.skip("requires the ab CLI on PATH")
    result = subprocess.run(
        [str(ab), "--output", "json", *args, "--source", str(project)],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "HOME": str(home), "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    assert result.returncode == 0, (
        f"exit {result.returncode}\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    return json.loads(result.stdout)


def test_tracing_list_and_get_read_the_local_dev_store(
    local_trace_store: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    home = tmp_path / "cli-home"
    home.mkdir()

    listed = _ab_json("tracing", "list", project=local_trace_store, home=home)
    assert listed, "`ab tracing list` returned no local traces"
    trace_id = listed[0]["trace_id"]
    assert isinstance(trace_id, str) and trace_id.startswith("tr-")

    got = _ab_json("tracing", "get", trace_id, project=local_trace_store, home=home)
    assert got["trace_id"] == trace_id
    assert got["status"]  # the probe span completed, so the trace has a status
