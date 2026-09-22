"""Unit tests for `mason dev`: wraps `databricks apps run-local` from the project dir."""

from __future__ import annotations

import pathlib
import types
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.cli import dev as dev_mod
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata


def _write_agent_manifest(
    source: pathlib.Path,
    *,
    server: str = "mason",
    memory: str | None = None,
    session: str | None = None,
) -> None:
    body = f'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "{server}"\n'
    if memory:
        body += f'\n[memory_store]\nname = "{memory}"\n'
    if session:
        body += f'\n[session_store]\nname = "{session}"\n'
    (source / "agent.toml").write_text(body)


class _Ctx:
    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile

    def client(self):
        return mock.Mock(current_user="me@example.com", host="https://my-workspace.databricks.com")


@pytest.fixture(autouse=True)
def _stub_local_tracing(monkeypatch):
    """`mason dev` starts a local MLflow tracking server (via cli.tracing) for Mason-server projects;
    stub the name dev.py imported so ordinary dev tests neither spawn one nor need uv. Tracing tests
    override this. The server helper's own behavior is tested in tracing_test.py."""
    monkeypatch.setattr(dev_mod, "start_local_tracing_server", lambda source_dir: (None, {}))


def test_dev_prepares_when_no_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # no .venv -> auto-prepare
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
    assert "--env" not in args[0]
    assert "--entry-point" in args[0]
    assert args[0][args[0].index("--entry-point") + 1] == "app.masondev.yaml"
    assert not (tmp_path / "app.masondev.yaml").exists()
    assert "--prepare-environment" in args[0]  # no venv yet -> build it
    assert args[1] == "ml"  # profile passed through
    assert kwargs["cwd"] == str(tmp_path)  # runs in the project dir


def test_dev_reuses_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()  # env already there -> don't rebuild
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" not in db.call_args.args[0]


def test_dev_force_prepare_overrides_existing_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--prepare-environment"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert "--prepare-environment" in db.call_args.args[0]  # explicit flag forces rebuild


def test_dev_no_prepare_and_custom_port(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev,
            ["--source", str(tmp_path), "--no-prepare-environment", "--app-port", "9000"],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    cmd = db.call_args.args[0]
    assert "--prepare-environment" not in cmd
    assert cmd[cmd.index("--app-port") : cmd.index("--app-port") + 2] == ["--app-port", "9000"]


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
    assert names == {"APP_SETTING", "DATABRICKS_MASON_RUNTIME_STORE_LOCAL"}


def test_dev_uses_local_entry_point_without_index_override(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump({"command": ["x"], "env": [{"name": "APP_SETTING", "value": "s"}]})
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    env = {e["name"]: e["value"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert env == {
        "APP_SETTING": "s",
        "DATABRICKS_MASON_RUNTIME_STORE_LOCAL": "true",
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
    # AGENT_SESSION_STORE). `mason dev` runs stores locally (memory off, sessions in-process), so the
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
    assert not (tmp_path / "app.masondev.yaml").exists()


def test_dev_does_not_wire_workspace_stores_when_bound(tmp_path: pathlib.Path, monkeypatch):
    # `mason dev` is a local sandbox: even with stores bound, it never wires the workspace store env.
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
        dev_yaml = pathlib.Path(kw["cwd"]) / "app.masondev.yaml"
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
    # For any project (here a custom-server one, to show it's not gated to the Mason server), dev
    # starts a local MLflow server and injects its MLFLOW_* env into the dev-only manifest
    # (app.masondev.yaml) — NOT into the deployable app.yaml — then tears the server down after the run.
    # The trace UI URL is announced.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(tmp_path, server="custom")
    (tmp_path / ".venv").mkdir()
    fake_server = mock.Mock()
    monkeypatch.setattr(
        dev_mod,
        "start_local_tracing_server",
        lambda source_dir: (
            fake_server,
            {
                "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
                "MLFLOW_EXPERIMENT_NAME": source_dir.resolve().name,
            },
        ),
    )
    captured: dict = {}

    def _fake_databricks(args, *a, **kw):
        # Read the dev manifest while it still exists (before finally-block cleanup).
        dev_yaml = pathlib.Path(kw["cwd"]) / "app.masondev.yaml"
        captured.update(yaml.safe_load(dev_yaml.read_text()))
        return types.SimpleNamespace(returncode=0)

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
    _write_agent_manifest(tmp_path, server="mason")
    (tmp_path / ".venv").mkdir()
    fake_server = mock.Mock()
    monkeypatch.setattr(
        dev_mod,
        "start_local_tracing_server",
        lambda source_dir: (fake_server, {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5599"}),
    )
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0  # the bad app.yaml surfaced as an error
    db.assert_not_called()  # failed before run-local was reached
    fake_server.terminate.assert_called_once()  # ...but the local server was still stopped


def test_dev_shows_local_traces_url_when_tracing_on(tmp_path: pathlib.Path, monkeypatch):
    # dev surfaces the local MLflow Traces URL plus the experiment name (in parens) so a dev run makes
    # clear where its traces land and which experiment to open.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(tmp_path, server="mason")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(
        dev_mod,
        "start_local_tracing_server",
        lambda source_dir: (
            mock.Mock(),
            {"MLFLOW_TRACKING_URI": "http://127.0.0.1:5599", "MLFLOW_EXPERIMENT_NAME": "my-agent"},
        ),
    )
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Traces" in output and "5599" in output
    assert "experiment name: my-agent" in output


def test_dev_omits_traces_line_when_local_tracing_unavailable(tmp_path: pathlib.Path, monkeypatch):
    # Local tracing couldn't start (start_local_tracing_server returns none) -> no Traces line in the panel.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(tmp_path, server="mason")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(dev_mod, "start_local_tracing_server", lambda source_dir: (None, {}))
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "Traces" not in result.output


def test_dev_runs_offline_when_client_unavailable(tmp_path: pathlib.Path):
    # No stores + no auth: obj.client() would raise, but with nothing bound it's never called (and
    # local tracing needs no workspace), so dev still runs the agent locally offline.
    from databricks_mason.errors import AgentCliError

    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()

    class _OfflineCtx:
        output = "text"
        profile = None

        def client(self):
            raise AgentCliError("no databricks auth configured")

    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_OfflineCtx())
    assert result.exit_code == 0, result.output
    assert db.call_args.args[0][:2] == ["apps", "run-local"]  # agent still ran


def test_dev_runs_without_traces_when_local_tracing_unavailable(
    tmp_path: pathlib.Path, monkeypatch
):
    # Local tracing is best-effort: if the server can't start (start_local_tracing_server returns none),
    # dev still runs the agent and injects no MLflow env.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(tmp_path, server="mason")
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(dev_mod, "start_local_tracing_server", lambda source_dir: (None, {}))
    captured: dict = {}

    def _fake_databricks(args, *a, **kw):
        captured.update(yaml.safe_load((pathlib.Path(kw["cwd"]) / "app.masondev.yaml").read_text()))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)
    result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert not any(e["name"].startswith("MLFLOW") for e in captured.get("env", []))


def test_dev_requires_app_yaml(tmp_path: pathlib.Path):
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "app.yaml" in result.output
    db.assert_not_called()


def test_dev_announces_chat_ui_when_overlay_present(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "ui.py").write_text("# chat UI\n")
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path), "--app-port", "9000"], obj=_Ctx()
        )
    assert result.exit_code == 0, result.output
    assert "Chat UI" in result.output
    assert "http://localhost:9000" in result.output
    output = " ".join(result.output.split())
    assert "mason endpoint invoke" in output
    assert "--url http://localhost:9000" in output
    assert "--path /api/invocations" in output
    assert "$(uuidgen)" in output


def test_dev_announces_api_endpoint_when_no_ui(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # API-only: no runtime/ui.py
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "API-only" in result.output
    assert "http://localhost:8000/invocations" in result.output
    # a copy-pasteable sample request, not just the bare endpoint
    assert "curl -X POST" in " ".join(result.output.split())
    output = " ".join(result.output.split())
    assert "mason endpoint invoke" in output
    assert "--url http://localhost:8000" in output
    assert "--path /invocations" in output


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
@pytest.mark.parametrize("server,chat_ui", [("mason", True), ("mason", False), ("custom", False)])
def test_dev_prints_standalone_invoke_for_each_template(tmp_path, framework, server, chat_ui):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(tmp_path, framework=framework, server=server).write()
    if chat_ui:
        (tmp_path / "runtime").mkdir()
        (tmp_path / "runtime" / "ui.py").write_text("# chat UI\n")
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    commands = [line for line in result.output.splitlines() if line.startswith("mason endpoint")]
    assert len(commands) == 1, result.output
    command = commands[0]
    path = "/api/invocations" if server == "mason" else "/invocations"
    assert f"mason endpoint invoke --url http://localhost:8000 --path {path} --json " in command
    assert "│" not in command
    assert ("$(uuidgen)" in command) is (server == "mason")
    panel, example = result.output.split("Invoke with Mason\n")
    assert panel.splitlines()[-1].startswith("╰")
    assert example.splitlines() == [command]
    assert any(line.startswith("│") and "mason deploy" in line for line in panel.splitlines())
    if server == "mason":
        assert any(
            line.startswith("│") and "mason tools add" in line for line in panel.splitlines()
        )
    if not chat_ui:
        assert "curl -X POST" in panel  # preserve the existing API-only next step


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_dev_custom_server_recommends_wiring_tools_in_agent_code(
    tmp_path: pathlib.Path,
    framework: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(tmp_path, framework=framework, server="custom").write()

    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "agent/agent.py" in output
    assert "mason tools add" not in output


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
        mock.patch.object(dev_mod, "_databricks") as db,
        mock.patch.object(ctx, "client") as client,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=ctx)

    assert result.exit_code != 0
    assert "require a Mason server template" in " ".join(result.output.split())
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
        mock.patch.object(dev_mod, "_databricks") as db,
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
        server="mason",
    ).write()
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "http://localhost:8000/api/invocations" in result.output


def test_dev_runs_from_project_containing_directly_edited_agent_manifest(
    tmp_path: pathlib.Path,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    manifest = tmp_path / "agent.toml"
    manifest.write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "mason"\n'
    )

    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert db.call_args.kwargs["cwd"] == str(tmp_path)
    assert manifest.read_text() == (
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\nserver = "mason"\n'
    )


def test_dev_warns_when_stores_unbound(tmp_path: pathlib.Path):
    # `mason dev` never provisions stores (unlike deploy); it warns so the gap isn't silent.
    (tmp_path / "app.yaml").write_text("command: []\n")  # no agent.toml -> both unbound
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" in result.output
    assert "No session store bound" in result.output


def test_dev_notes_local_stores_when_bound(tmp_path: pathlib.Path):
    # With stores bound, dev doesn't warn "not bound"; it notes they're used once deployed while dev
    # runs them locally. Dev makes no workspace call (the _Ctx client would fail the run if used).
    (tmp_path / "app.yaml").write_text("command: []\n")
    _write_agent_manifest(tmp_path, memory="mem", session="sess")
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" not in result.output
    assert "No session store bound" not in result.output
    assert "used once deployed" in result.output  # the local-sandbox note
