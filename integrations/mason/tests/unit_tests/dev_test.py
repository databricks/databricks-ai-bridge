"""Unit tests for `mason dev`: wraps `databricks apps run-local` from the project dir."""

from __future__ import annotations

import pathlib
from unittest import mock

from click.testing import CliRunner

from databricks_mason import dev as dev_mod


class _Ctx:
    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile
        self.client_calls = 0  # so tests can assert dev stays auth-free when nothing needs it

    def client(self):
        # Constructing the client / reading current_user is a workspace (auth) call; dev must only
        # do it when tracing or a store is actually configured.
        self.client_calls += 1
        return mock.Mock(current_user="me@example.com")


def test_dev_prepares_when_no_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # no .venv -> auto-prepare
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
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
    assert cmd[-2:] == ["--app-port", "9000"]


def test_dev_filters_build_index_env_via_entry_point(tmp_path: pathlib.Path):
    import yaml

    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    {"name": "AGENT_SESSION_STORE", "value": "s"},
                    {"name": "PIP_INDEX_URL", "value": "https://pypi.org/simple/"},
                    {"name": "UV_INDEX_URL", "value": "https://pypi.org/simple/"},
                ],
            }
        )
    )
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    cmd = db.call_args.args[0]
    assert "--entry-point" in cmd  # a filtered manifest was used
    dev_yaml = tmp_path / ".mason-dev.app.yaml"
    assert str(dev_yaml) in cmd
    names = {e["name"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert "PIP_INDEX_URL" not in names and "UV_INDEX_URL" not in names  # index vars stripped
    assert "AGENT_SESSION_STORE" in names  # app env kept
    assert not any(n.startswith("MLFLOW") for n in names)  # tracing not configured -> not wired


def test_dev_no_entry_point_when_no_index_override(tmp_path: pathlib.Path):
    import yaml

    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump({"command": ["x"], "env": [{"name": "AGENT_SESSION_STORE", "value": "s"}]})
    )
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "--entry-point" not in db.call_args.args[0]  # nothing to strip -> use app.yaml as-is
    assert not (tmp_path / ".mason-dev.app.yaml").exists()


def test_dev_with_flags_wires_app_yaml_before_running(tmp_path: pathlib.Path):
    import yaml

    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()
    with (
        mock.patch.object(dev_mod, "_databricks") as db,
        mock.patch.object(
            dev_mod,
            "resolve_store_env",
            return_value={"AGENT_SESSION_STORE": "s", "AGENT_MEMORY_STORE": "abc"},
        ) as resolve,
    ):
        result = CliRunner().invoke(
            dev_mod.dev,
            ["--source", str(tmp_path), "--session", "s", "--memory", "m"],
            obj=_Ctx(),
        )
    assert result.exit_code == 0, result.output
    resolve.assert_called_once()  # same resolution path as deploy
    env = {
        e["name"]: e["value"] for e in yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    }
    assert env["AGENT_SESSION_STORE"] == "s"  # patched before run
    assert env["AGENT_MEMORY_STORE"] == "abc"
    assert not any(k.startswith("MLFLOW") for k in env)  # tracing not configured here
    assert db.call_args.args[0][:2] == ["apps", "run-local"]


def test_dev_wires_no_tracing_when_unconfigured(tmp_path: pathlib.Path):
    # Tracing is opt-in + UC-only: with no `mason tracing setup`, dev wires no MLFLOW env at all -
    # AND never touches the workspace client (no auth/`me()` call), so offline local dev works.
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    ctx = _Ctx()
    with (
        mock.patch.object(dev_mod, "_databricks"),
        mock.patch.object(dev_mod, "resolve_store_env") as resolve,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=ctx)
    assert result.exit_code == 0, result.output
    resolve.assert_not_called()  # no --memory/--session -> no store resolution
    assert ctx.client_calls == 0  # unconfigured dev makes no workspace/auth call
    import yaml

    doc = yaml.safe_load((tmp_path / "app.yaml").read_text()) or {}
    env = {e["name"]: e["value"] for e in (doc.get("env") or [])}
    assert not any(k.startswith("MLFLOW") for k in env)
    assert not any(k.startswith("AGENT_") for k in env)


def test_dev_wires_tracing_through_shared_provision(tmp_path: pathlib.Path):
    # When tracing IS configured (agent.toml trace_location), dev wires it via the exact same
    # provision_trace_experiment path as deploy (the actual UC work is covered by deploy_test).
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n\n[trace_location]\nname = "cat.schema"\n'
    )
    with (
        mock.patch.object(dev_mod, "_databricks"),
        mock.patch.object(
            dev_mod,
            "provision_trace_experiment",
            return_value=("/Users/me@example.com/mason-traces/x", "cat.schema"),
        ) as prov,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    prov.assert_called_once()  # configured -> tracing wired through the shared path
    assert prov.call_args.args[1] == tmp_path.resolve().name  # app name
    assert prov.call_args.args[2] == "me@example.com"  # current_user (only read once configured)


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


def test_dev_announces_api_endpoint_when_no_ui(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # API-only: no runtime/ui.py
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "API-only" in result.output
    assert "http://localhost:8000/invocations" in result.output
    # a copy-pasteable sample request, not just the bare endpoint
    assert "curl -X POST" in " ".join(result.output.split())


def test_dev_runs_from_project_containing_directly_edited_agent_manifest(
    tmp_path: pathlib.Path,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    manifest = tmp_path / "agent.toml"
    manifest.write_text('schema_version = 1\n\n[agent]\nframework = "langgraph"\n')

    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert db.call_args.kwargs["cwd"] == str(tmp_path)
    assert manifest.read_text() == 'schema_version = 1\n\n[agent]\nframework = "langgraph"\n'
