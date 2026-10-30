"""Unit tests for `agentbricks dev`: wraps `databricks apps run-local` from the project dir."""

from __future__ import annotations

import pathlib
import subprocess
import types
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from databricks_agentbricks.cli import dev as dev_mod
from databricks_agentbricks.cli.app import CliContext
from databricks_agentbricks.clients.local_tracing_client import LocalTracingStart
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.endpoint import agent_invoke_command
from databricks_agentbricks.projects.agent_project import AgentProject, ToolSpec
from databricks_agentbricks.projects.config import write_project_metadata


def _write_agent_manifest(
    source: pathlib.Path,
    *,
    server: str = "agentbricks",
    memory: str | None = None,
    session: str | None = None,
    experiment_name: str | None = None,
) -> None:
    body = f'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "{server}"\n'
    if memory:
        body += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        body += f'\n[session_store]\nname = "{session}"\n'
    if experiment_name:
        body += f'\n[tracing]\nexperiment_name = "{experiment_name}"\n'
    (source / "agent.toml").write_text(body)


class _Ctx(CliContext):
    def __init__(self, output: str = "text", profile=None):
        super().__init__(profile, output)
        self.api_client_provider = types.SimpleNamespace(
            get=lambda: mock.Mock(
                current_user="me@example.com", host="https://my-workspace.databricks.com"
            )
        )

    def client(self):
        return mock.Mock(current_user="me@example.com", host="https://my-workspace.databricks.com")


def _run_local_starts(args, profile, *, on_ready=None, **kwargs):
    """Fake `databricks apps run-local`: the agent comes up, then exits cleanly (e.g. Ctrl+C)."""
    if on_ready is not None:
        on_ready()
    return subprocess.CompletedProcess(["databricks", *args], 0)


def _panel_text(output: str) -> str:
    """The success panel's rows joined back into text, without borders or padding."""
    return " ".join(line.strip("│ ") for line in output.splitlines() if line.startswith("│"))


@pytest.fixture(autouse=True)
def _authenticated_profile(validated_profile):
    """Preflight validation is the workspace boundary; these tests are not about workspace_auth."""


@pytest.fixture(autouse=True)
def _stub_local_tracing(monkeypatch):
    """`agentbricks dev` starts a local MLflow tracking server for Agent Bricks projects;
    stub the client so ordinary dev tests neither spawn one nor need uv. Tracing tests override
    this. The client's own behavior is tested in tracing_test.py."""
    monkeypatch.setattr(
        dev_mod.LocalTracingClient,
        "start_dev",
        lambda _self, source_dir: LocalTracingStart(None, None),
    )


@pytest.fixture(autouse=True)
def _stub_free_port(monkeypatch):
    """Treat the app port as free so dev tests never depend on what this machine is running."""
    monkeypatch.setattr(dev_mod, "port_in_use", lambda port: False)
    monkeypatch.setattr(dev_mod, "free_port", lambda: 54321)


def test_dev_prepares_when_no_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # no .venv -> auto-prepare
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
    assert "--env" not in args[0]
    assert "--entry-point" in args[0]
    assert args[0][args[0].index("--entry-point") + 1] == "app.agentbricksdev.yaml"
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    assert "--prepare-environment" in args[0]  # no venv yet -> build it
    assert args[1] == "ml"  # profile passed through
    assert kwargs["cwd"] == str(tmp_path)  # runs in the project dir


def test_dev_reuses_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()  # env already there -> don't rebuild
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" not in db.call_args.args[0]


def test_dev_force_prepare_overrides_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--prepare-environment"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" in db.call_args.args[0]  # explicit flag forces rebuild


def test_dev_no_prepare_and_custom_port(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(
            dev_mod.dev,
            ["--source", str(tmp_path), "--no-prepare-environment", "--app-port", "9000"],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    cmd = db.call_args.args[0]
    assert "--prepare-environment" not in cmd
    # The selected port is run-local's proxy; the app runs behind it on a free port.
    assert cmd[cmd.index("--port") : cmd.index("--port") + 2] == ["--port", "9000"]
    assert cmd[cmd.index("--app-port") : cmd.index("--app-port") + 2] == ["--app-port", "54321"]


def test_dev_default_port_never_leaves_run_local_on_its_fixed_proxy_port(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    cmd = db.call_args.args[0]
    assert cmd[cmd.index("--port") : cmd.index("--port") + 2] == ["--port", "8000"]
    assert cmd[cmd.index("--app-port") : cmd.index("--app-port") + 2] == ["--app-port", "54321"]
    assert "8001" not in cmd


def test_dev_filters_build_index_env_via_entry_point(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    {"name": "APP_SETTING", "value": "s"},
                    {"name": "PIP_INDEX_URL", "value": "https://pypi.org/simple/"},
                    {"name": "UV_INDEX_URL", "value": "https://pypi.org/simple/"},
                ],
            }
        )
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    names = {e["name"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert names == {"APP_SETTING", "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL"}


def test_dev_uses_local_entry_point_without_index_override(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump({"command": ["x"], "env": [{"name": "APP_SETTING", "value": "s"}]})
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    env = {e["name"]: e["value"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert env == {
        "APP_SETTING": "s",
        "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL": "true",
    }
    original_env = yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    assert original_env == [{"name": "APP_SETTING", "value": "s"}]


def test_dev_entry_point_strips_inherited_workspace_tracing_env(tmp_path: pathlib.Path):
    # A previously-deployed app.yaml carries workspace tracing env (MLFLOW_TRACKING_URI + a workspace
    # MLFLOW_EXPERIMENT_ID). The dev manifest must NOT inherit it — a stale id, which MLflow resolves
    # ahead of MLFLOW_EXPERIMENT_NAME, would point local tracing at an id absent from the local store.
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
                    {"name": "MLFLOW_EXPERIMENT_ID", "value": "999"},
                    {"name": "APP_SETTING", "value": "keep"},
                ],
            }
        )
    )
    dev_yaml = dev_mod._dev_entry_point(
        tmp_path / "app.yaml",
        {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5599", "MLFLOW_EXPERIMENT_NAME": "my-agent"},
    )
    env = {e["name"]: e["value"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert (
        "MLFLOW_EXPERIMENT_ID" not in env
    )  # stale workspace id stripped, so it can't win over NAME
    assert env["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5599"  # local server wins
    assert env["MLFLOW_EXPERIMENT_NAME"] == "my-agent"
    assert env["APP_SETTING"] == "keep"  # unrelated env preserved


def test_dev_strips_inherited_deploy_store_env(tmp_path: pathlib.Path):
    # A previously-deployed app.yaml carries the workspace store env (AGENT_MEMORY_STORE +
    # AGENT_SESSION_STORE). `agentbricks dev` runs stores locally (memory off, sessions in-process), so the
    # dev manifest must NOT inherit them - otherwise dev would silently use the workspace stores.
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    {"name": "AGENT_MEMORY_STORE", "value": "mem-id"},
                    {"name": "AGENT_SESSION_STORE", "value": "sess"},
                    {"name": "APP_SETTING", "value": "keep"},
                ],
            }
        )
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    env = {e["name"]: e["value"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert "AGENT_MEMORY_STORE" not in env
    assert "AGENT_SESSION_STORE" not in env
    assert env["APP_SETTING"] == "keep"  # unrelated env preserved


def test_dev_entry_point_rejects_non_list_env(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("env:\n  KEY: value\n")

    with pytest.raises(AgentCliError, match="env must be a list"):
        dev_mod._dev_entry_point(tmp_path / "app.yaml")


def test_dev_entry_point_rejects_non_object_manifest(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("- command\n- uv\n")

    with pytest.raises(AgentCliError, match="top level must be an object"):
        dev_mod._dev_entry_point(tmp_path / "app.yaml")


def test_dev_removes_local_entry_point_when_run_local_fails(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=RuntimeError("failed")):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()


def test_dev_does_not_wire_workspace_stores_when_bound(tmp_path: pathlib.Path, monkeypatch):
    # `agentbricks dev` is a local sandbox: even with stores bound, it never wires the workspace store env.
    # The dev-only manifest carries no AGENT_MEMORY_STORE / AGENT_SESSION_STORE (the runtime falls back
    # to memory-off / in-process sessions), and the deployable app.yaml stays clean (deploy owns that).
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(src, memory="m", session="s")
    (src / ".venv").mkdir()
    captured_dev: dict = {}

    def _fake_databricks(args, *a, **kw):
        # Read the dev manifest while it still exists (before finally-block cleanup).
        dev_yaml = pathlib.Path(kw["cwd"]) / "app.agentbricksdev.yaml"
        captured_dev.update(yaml.safe_load(dev_yaml.read_text()))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)
    result = CliRunner().invoke(dev_mod.dev, ["--source", str(src)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    dev_env = {e["name"] for e in captured_dev.get("env", [])}
    assert "AGENT_MEMORY_STORE" not in dev_env and "AGENT_SESSION_STORE" not in dev_env
    # deployable app.yaml untouched
    assert (yaml.safe_load((src / "app.yaml").read_text()).get("env") or []) == []


def test_dev_starts_local_tracing_and_wires_dev_manifest(tmp_path: pathlib.Path, monkeypatch):
    # For any project (here a custom-server one, to show it's not gated to the AgentBricks server), dev
    # starts a local MLflow server and injects its MLFLOW_* env into the dev-only manifest
    # (app.agentbricksdev.yaml) — NOT into the deployable app.yaml — then tears the server down after the run.
    # The trace UI URL is announced.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(tmp_path, server="custom")
    (tmp_path / ".venv").mkdir()
    fake_server = mock.Mock()
    monkeypatch.setattr(
        dev_mod.LocalTracingClient,
        "start_dev",
        lambda _self, source_dir: LocalTracingStart(
            fake_server,
            "http://127.0.0.1:5599",
            environment={
                "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
                "MLFLOW_EXPERIMENT_NAME": source_dir.resolve().name,
            },
        ),
    )
    captured: dict = {}

    def _fake_databricks(args, *a, **kw):
        # Read the dev manifest while it still exists (before finally-block cleanup).
        dev_yaml = pathlib.Path(kw["cwd"]) / "app.agentbricksdev.yaml"
        captured.update(yaml.safe_load(dev_yaml.read_text()))
        return _run_local_starts(args, *a, **kw)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)
    result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    dev_env = {e["name"]: e["value"] for e in captured.get("env", [])}
    assert dev_env["MLFLOW_TRACKING_URI"] == "http://127.0.0.1:5599"
    assert dev_env["MLFLOW_EXPERIMENT_NAME"] == tmp_path.resolve().name
    # The deployable app.yaml stays clean of MLflow env — only the dev manifest carries it.
    real_env = {
        e["name"]: e["value"]
        for e in (yaml.safe_load((tmp_path / "app.yaml").read_text()).get("env") or [])
    }
    assert not any(name.startswith("MLFLOW") for name in real_env)
    assert "5599" in result.output  # trace UI announced
    fake_server.terminate.assert_called_once()  # torn down after the run


def test_dev_stops_local_tracing_server_when_setup_fails(tmp_path: pathlib.Path, monkeypatch):
    # A failure after the local server starts but before run-local (here a malformed app.yaml that
    # `_dev_entry_point` rejects) must still tear the server down, not orphan it — everything after the
    # server start runs under the try/finally.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": "not-a-list"}))
    _write_agent_manifest(tmp_path, server="agentbricks")
    (tmp_path / ".venv").mkdir()
    fake_server = mock.Mock()
    monkeypatch.setattr(
        dev_mod.LocalTracingClient,
        "start_dev",
        lambda _self, source_dir: LocalTracingStart(
            fake_server,
            "http://127.0.0.1:5599",
            environment={"MLFLOW_TRACKING_URI": "http://127.0.0.1:5599"},
        ),
    )
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0  # the bad app.yaml surfaced as an error
    db.assert_not_called()  # failed before run-local was reached
    fake_server.terminate.assert_called_once()  # ...but the local server was still stopped


def test_dev_shows_local_traces_url_when_tracing_on(tmp_path: pathlib.Path, monkeypatch):
    # dev surfaces the local MLflow Traces URL plus the experiment name (in parens) so a dev run makes
    # clear where its traces land and which experiment to open.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(tmp_path, server="agentbricks")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(
        dev_mod.LocalTracingClient,
        "start_dev",
        lambda _self, source_dir: LocalTracingStart(
            mock.Mock(),
            "http://127.0.0.1:5599",
            environment={
                "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
                "MLFLOW_EXPERIMENT_NAME": "my-agent",
            },
        ),
    )
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Traces" in output and "5599" in output
    assert "experiment name: my-agent" in output


def test_dev_omits_traces_line_when_local_tracing_unavailable(tmp_path: pathlib.Path):
    # Local tracing couldn't start, so there is no Traces line in the panel.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(tmp_path, server="agentbricks")
    (tmp_path / ".venv").mkdir()
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "Traces" not in result.output


def test_dev_runs_offline_when_client_unavailable(tmp_path: pathlib.Path):
    # No stores + no auth: obj.client() would raise, but with nothing bound it's never called (and
    # local tracing needs no workspace), so dev still runs the agent locally offline.
    from databricks_agentbricks.errors import AgentCliError

    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()

    class _OfflineCtx(CliContext):
        def __init__(self):
            super().__init__(None, "text")
            self.api_client_provider = types.SimpleNamespace(
                get=lambda: (_ for _ in ()).throw(AgentCliError("no databricks auth configured"))
            )

        def client(self):
            raise AgentCliError("no databricks auth configured")

    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_OfflineCtx())
    assert result.exit_code == 0, result.output
    assert db.call_args.args[0][:2] == ["apps", "run-local"]  # agent still ran


def test_dev_runs_without_traces_when_local_tracing_unavailable(
    tmp_path: pathlib.Path, monkeypatch
):
    # Local tracing is best-effort: if the server can't start,
    # dev still runs the agent and injects no MLflow env.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(tmp_path, server="agentbricks")
    (tmp_path / ".venv").mkdir()
    captured: dict = {}

    def _fake_databricks(args, *a, **kw):
        captured.update(
            yaml.safe_load((pathlib.Path(kw["cwd"]) / "app.agentbricksdev.yaml").read_text())
        )
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)
    result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert not any(e["name"].startswith("MLFLOW") for e in captured.get("env", []))


def test_dev_requires_app_yaml(tmp_path: pathlib.Path):
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "app.yaml" in result.output
    db.assert_not_called()


def test_dev_announces_chat_ui_when_overlay_present(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "ui.py").write_text("# chat UI\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--app-port", "9000"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert "Chat UI" in result.output
    assert "http://localhost:9000" in result.output
    output = _panel_text(result.output)
    assert "agentbricks endpoint invoke" in output
    assert "--url http://localhost:9000" in output
    assert "--path /api/invocations" in output
    assert "$(uuidgen)" in output


def test_dev_announces_api_endpoint_when_no_ui(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # API-only: no runtime/ui.py
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "API-only" in result.output
    assert "http://localhost:8000/invocations" in result.output
    # a copy-pasteable sample request, not just the bare endpoint
    assert "curl -X POST" in " ".join(result.output.split())
    output = _panel_text(result.output)
    assert "agentbricks endpoint invoke" in output
    assert "--url http://localhost:8000" in output
    assert "--path /invocations" in output


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize(
    "server,chat_ui", [("agentbricks", True), ("agentbricks", False), ("custom", False)]
)
def test_dev_prints_invoke_inside_panel_for_each_template(tmp_path, framework, server, chat_ui):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(tmp_path, framework=framework, server=server).write()
    if chat_ui:
        (tmp_path / "runtime").mkdir()
        (tmp_path / "runtime" / "ui.py").write_text("# chat UI\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    # The success output ends at the panel's bottom border; nothing trails it.
    assert [line for line in lines if line.strip()][-1].startswith("╰"), result.output
    label = next(i for i, line in enumerate(lines) if "Invoke with Agent Bricks" in line)
    deploy = next(i for i, line in enumerate(lines) if "agentbricks deploy" in line)
    assert lines[label].startswith("│") and lines[deploy].startswith("│")
    assert deploy < label  # the invoke example is the last entry, after the other next steps
    rows = [line.strip("│ ") for line in lines[label + 1 :] if line.startswith("│")]
    assert all(line.startswith(("│", "╰")) for line in lines[label:] if line.strip())
    command = agent_invoke_command(
        "--url http://localhost:8000", uses_runtime_api=server == "agentbricks"
    )
    assert "".join("".join(rows).split()) == "".join(command.split()), result.output
    path = "/api/invocations" if server == "agentbricks" else "/invocations"
    assert f"--path {path} --json " in command
    assert ("$(uuidgen)" in command) is (server == "agentbricks")
    panel = "\n".join(lines[:label])
    if server == "agentbricks":
        assert "agentbricks tools add" in panel
    if not chat_ui:
        assert "curl -X POST" in panel  # preserve the existing API-only next step


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_dev_custom_server_recommends_wiring_tools_in_agent_code(
    tmp_path: pathlib.Path,
    framework: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(tmp_path, framework=framework, server="custom").write()

    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "agent/agent.py" in output
    assert "agentbricks tools add" not in output


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
def test_dev_rejects_custom_server_manifest_tools_before_starting(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    project = AgentProject.create(tmp_path, framework=framework, server="custom")
    project.add_tool(ToolSpec.mcp("web", service="system.ai.web_search"))
    project.write()
    write_project_metadata(tmp_path, framework=framework, template=template)
    manifest = tmp_path / "agent.toml"
    before = manifest.read_text(encoding="utf-8")
    ctx = _Ctx()

    with (
        mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db,
        mock.patch.object(ctx, "client") as client,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=ctx)

    assert result.exit_code != 0
    assert "require an Agent Bricks server template" in " ".join(result.output.split())
    assert manifest.read_text(encoding="utf-8") == before
    client.assert_not_called()
    db.assert_not_called()


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_dev_surfaces_invalid_custom_server_manifest_before_starting(
    tmp_path: pathlib.Path,
    framework: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "agent.toml").write_text(
        f'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "custom"\n'
        '\n[[tools]]\nid = "legacy"\nsource = { kind = "python", '
        'entrypoint = "agent.tools:legacy" }\n',
        encoding="utf-8",
    )
    ctx = _Ctx()

    with (
        mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db,
        mock.patch.object(ctx, "client") as client,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=ctx)

    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "Python tools are code-first" in output
    assert "framework-native agent code" in output
    assert "remain active" not in output
    client.assert_not_called()
    db.assert_not_called()


def test_dev_standard_template_uses_runtime_api(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(
        tmp_path,
        framework="langgraph",
        server="agentbricks",
    ).write()
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "http://localhost:8000/api/invocations" in result.output


def test_dev_runs_from_project_containing_directly_edited_agent_manifest(
    tmp_path: pathlib.Path,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    manifest = tmp_path / "agent.toml"
    manifest.write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
    )

    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert db.call_args.kwargs["cwd"] == str(tmp_path)
    assert manifest.read_text() == (
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "agentbricks"\n'
    )


def test_dev_quiet_about_unbound_stores(tmp_path: pathlib.Path):
    # Unbound stores are a `agentbricks deploy` concern; `agentbricks dev` doesn't warn about them, it just runs
    # locally (memory off, sessions in-process).
    (tmp_path / "app.yaml").write_text("command: []\n")  # no agent.toml -> both unbound
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" not in result.output
    assert "No session store bound" not in result.output
    assert "Tracing experiment" not in result.output  # unbound tracing -> no notice


def test_dev_notes_local_stores_when_bound(tmp_path: pathlib.Path):
    # With stores bound, dev notes they're bound but run locally here (used only once deployed). Dev
    # makes no workspace call (the _Ctx client would fail the run if used).
    (tmp_path / "app.yaml").write_text("command: []\n")
    _write_agent_manifest(tmp_path, memory="mem", session="sess")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())  # collapse rich line-wrapping
    assert "Memory store 'mem' is bound" in out
    assert "Session store 'sess' is bound" in out
    assert "Run `agentbricks deploy` to use bound store" in out


def test_dev_notes_bound_tracing_experiment(tmp_path: pathlib.Path):
    # A bound tracing experiment is a `agentbricks deploy` concern; dev always traces to its own local MLflow
    # server, so it notes the binding but explains the deployed agent is what uses it (mirrors the
    # memory/session store notices). Dev makes no workspace call.
    (tmp_path / "app.yaml").write_text("command: []\n")
    _write_agent_manifest(tmp_path, experiment_name="/Shared/agentbricks_traces/mine")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())  # collapse rich line-wrapping
    assert "Tracing experiment '/Shared/agentbricks_traces/mine' is bound" in out
    assert "Run `agentbricks deploy` to trace to the bound experiment" in out


def test_dev_reports_local_tracing_start_warning(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    failure = LocalTracingStart(None, None, warning="uvx not found", help="install uv")
    with (
        mock.patch.object(dev_mod.LocalTracingClient, "start_dev", return_value=failure),
        mock.patch.object(dev_mod.render, "diagnostic") as diagnostic,
        mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    diagnostic.assert_called_once_with("warning", "uvx not found", help="install uv")
    db.assert_called_once()


def test_dev_hides_active_venv_and_port_from_run_local(tmp_path: pathlib.Path, monkeypatch):
    # run-local's `uv pip install` targets $VIRTUAL_ENV, so an activated venv would get the app's packages;
    # an exported PORT would keep run-local from pointing the app at --app-port.
    (tmp_path / "app.yaml").write_text("command: []\n")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/.venv")
    monkeypatch.setenv("PORT", "6969")
    monkeypatch.setenv("AGENTBRICKS_DEV_TEST_MARKER", "kept")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    env = db.call_args.kwargs["env"]
    assert "VIRTUAL_ENV" not in env
    assert "PORT" not in env
    assert env["AGENTBRICKS_DEV_TEST_MARKER"] == "kept"


def _write_project(source: pathlib.Path, env_file: str) -> None:
    (source / "app.yaml").write_text("command: []\n")
    (source / ".venv").mkdir()
    (source / ".env").write_text(env_file)


def _run_dev(source: pathlib.Path, monkeypatch, env_file: str, profile=None):
    """Run `dev` on a fresh project; returns (result, run-local profiles, local-manifest env)."""
    _write_project(source, env_file)
    profiles: list = []
    manifest_env: dict[str, str] = {}

    def _fake_databricks(cmd, profile, **kwargs):
        profiles.append(profile)
        manifest = yaml.safe_load(
            (pathlib.Path(kwargs["cwd"]) / "app.agentbricksdev.yaml").read_text()
        )
        manifest_env.update({e["name"]: e["value"] for e in manifest["env"]})
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)
    result = CliRunner().invoke(dev_mod.dev, ["--source", str(source)], obj=_Ctx(profile=profile))
    return result, profiles, manifest_env


@pytest.mark.parametrize(
    ("flag_profile", "expected", "note"),
    [
        (None, "proj", "Using profile 'proj' (from .env) · https://ws.example"),
        ("flag", "flag", "will run with profile 'flag' (from --profile), not .env's 'proj'"),
    ],
)
def test_dev_runs_with_resolved_profile(
    tmp_path, monkeypatch, validated_profile, write_databrickscfg, flag_profile, expected, note
):
    write_databrickscfg("[proj]\nhost = https://proj.example\n")
    result, profiles, manifest_env = _run_dev(
        tmp_path, monkeypatch, "DATABRICKS_CONFIG_PROFILE=proj\n", profile=flag_profile
    )

    assert result.exit_code == 0, result.output
    assert profiles == [expected]
    assert manifest_env["DATABRICKS_CONFIG_PROFILE"] == expected
    validated_profile.assert_called_once_with(expected)
    assert note in " ".join(result.output.split())


# A client id without its secret can't authenticate on its own, so the profile still applies.
@pytest.mark.parametrize(
    ("credentials", "uses_profile"),
    [
        ("DATABRICKS_TOKEN=dapi123\n", False),
        ("DATABRICKS_CLIENT_ID=sp\nDATABRICKS_CLIENT_SECRET=secret\n", False),
        ("DATABRICKS_CLIENT_ID=sp\n", True),
    ],
)
def test_dev_env_credentials_replace_the_profile_only_when_complete(
    tmp_path, monkeypatch, validated_profile, credentials, uses_profile
):
    env_file = f"DATABRICKS_CONFIG_PROFILE=proj\n{credentials}"
    result, _, manifest_env = _run_dev(tmp_path, monkeypatch, env_file)

    assert result.exit_code == 0, result.output
    assert manifest_env.get("DATABRICKS_CONFIG_PROFILE") == ("proj" if uses_profile else None)
    assert validated_profile.call_args_list == ([mock.call("proj")] if uses_profile else [])


def test_dev_unauthenticated_profile_fails_before_run_local(
    tmp_path: pathlib.Path, monkeypatch, fail_validation
):
    _write_project(tmp_path, "DATABRICKS_CONFIG_PROFILE=proj\n")
    fail_validation(RuntimeError("expired"))
    run_local = mock.Mock()
    monkeypatch.setattr(dev_mod, "_databricks", run_local)

    result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    out = " ".join(result.output.split())
    assert result.exit_code != 0
    assert "Using profile 'proj' (from .env)" in out
    assert "'proj' (from .env) isn't authenticated" in out
    assert "agentbricks auth login proj" in out
    run_local.assert_not_called()


def test_dev_busy_port_renders_error_and_hint(tmp_path: pathlib.Path, monkeypatch):
    (tmp_path / "app.yaml").write_text("command: []\n")
    monkeypatch.setattr(dev_mod, "port_in_use", lambda port: port == 9100)
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts) as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--app-port", "9100"], obj=_Ctx()
        )
    assert result.exit_code != 0
    output = " ".join(result.output.split())
    assert "Port 9100 is already in use (possibly a previous agentbricks dev run)." in output
    assert "agentbricks dev --app-port <port>" in output
    db.assert_not_called()
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()


def test_dev_says_starting_before_the_agent_is_ready(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks", side_effect=_run_local_starts):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    starting = result.output.index("Starting agent locally")
    assert starting < result.output.index("is running")  # neutral line first, success panel after


def test_dev_hides_success_panel_when_agent_fails_to_start(tmp_path: pathlib.Path):
    # run-local exits non-zero (e.g. no ~/.databrickscfg) before the agent ever answers on its port:
    # no green "running" panel, just the starting line and a red error pointing at the output above.
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "ui.py").write_text("# chat UI\n")

    def _fails(args, profile, **kwargs):
        return subprocess.CompletedProcess(["databricks", *args], 1)

    with mock.patch.object(dev_mod, "_databricks", side_effect=_fails):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "Starting agent locally" in result.output
    assert "is running" not in result.output
    assert "✔" not in result.output and "Chat UI" not in result.output
    assert "Could not start the agent locally." in result.output
    assert "See the output above for the cause." in result.output
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    # Blank lines set our status and error apart from the run-local output between them.
    lines = result.output.splitlines()
    starting = next(i for i, line in enumerate(lines) if "Starting agent locally" in line)
    error = next(i for i, line in enumerate(lines) if "Could not start" in line)
    assert lines[starting - 1] == "" and lines[starting + 1] == ""
    assert lines[error - 1] == ""


def test_dev_reports_agent_that_stops_after_starting(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")

    def _starts_then_crashes(args, profile, *, on_ready, **kwargs):
        on_ready()
        return subprocess.CompletedProcess(["databricks", *args], 1)

    with mock.patch.object(dev_mod, "_databricks", side_effect=_starts_then_crashes):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "is running" in result.output
    assert "The local agent stopped unexpectedly." in result.output
    assert "Could not start" not in result.output


def test_dev_waits_for_the_agent_port(tmp_path: pathlib.Path, monkeypatch):
    # Readiness is probed on the agent's own (free) port, not run-local's proxy port: the proxy on
    # --app-port listens before the agent is up, so probing it would show success too early.
    (tmp_path / "app.yaml").write_text("command: []\n")
    probed: list[int] = []
    monkeypatch.setattr(
        "databricks_agentbricks.clients.apps_client.accepts_connections",
        lambda port: probed.append(port) or True,
    )

    def _probes(args, profile, *, ready_probe, on_ready, **kwargs):
        assert ready_probe()
        on_ready()
        return subprocess.CompletedProcess(["databricks", *args], 0)

    with mock.patch.object(dev_mod, "_databricks", side_effect=_probes):
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--app-port", "9000"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert probed == [54321]  # the stubbed free port the agent serves on, behind the proxy on 9000
