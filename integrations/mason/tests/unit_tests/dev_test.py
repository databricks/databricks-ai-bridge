"""Unit tests for `mason dev`: wraps `databricks apps run-local` from the project dir."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_mason import dev as dev_mod
from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.project_config import write_project_metadata


class _Ctx:
    def __init__(self, output: str = "text", profile=None):
        self.output = output
        self.profile = profile

    def client(self):
        return mock.Mock(current_user="me@example.com")


@pytest.fixture(autouse=True)
def _stub_tracing(monkeypatch):
    """Tracing is on by default and would hit MLflow/the workspace; stub the provisioning so the
    non-tracing dev tests stay hermetic. Tracing-specific tests override this."""
    monkeypatch.setattr(dev_mod, "resolve_trace_experiment_id", lambda *a, **k: None)


def test_dev_prepares_when_no_venv(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")  # no .venv -> auto-prepare
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(
            dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx(profile="ml")
        )
    assert result.exit_code == 0, result.output
    args, kwargs = db.call_args
    assert args[0][:2] == ["apps", "run-local"]
    assert "DATABRICKS_MASON_RUNTIME_LOCAL=true" in args[0]
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
    assert names == {"AGENT_SESSION_STORE"}  # index vars stripped, app env kept


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


def test_dev_validates_bound_stores_without_writing_store_env(tmp_path: pathlib.Path):
    import yaml

    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "m"\n\n[session_store]\nname = "s"\n',
        encoding="utf-8",
    )
    (tmp_path / ".venv").mkdir()
    with (
        mock.patch.object(dev_mod, "_databricks") as db,
        mock.patch.object(dev_mod, "validate_stores") as validate,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    validate.assert_called_once()  # bound stores are validated, same path as deploy
    # Stores are read from agent.toml at runtime, so no store env is written into app.yaml.
    env_entries = yaml.safe_load((tmp_path / "app.yaml").read_text()).get("env") or []
    assert {e["name"] for e in env_entries} == set()
    assert db.call_args.args[0][:2] == ["apps", "run-local"]


def test_dev_without_bindings_does_not_validate(tmp_path: pathlib.Path):
    # No agent.toml store bindings -> nothing to validate (tracing is stubbed by the autouse fixture).
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    with (
        mock.patch.object(dev_mod, "_databricks"),
        mock.patch.object(dev_mod, "validate_stores") as validate,
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    validate.assert_not_called()


def test_dev_wires_tracing_env_on_by_default(tmp_path: pathlib.Path, monkeypatch):
    # Tracing is on by default: dev resolves the per-project experiment and wires the two MLflow env vars
    # into app.yaml (the experiment id + the workspace tracking uri).
    import yaml

    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(dev_mod, "resolve_trace_experiment_id", lambda *a, **k: "exp-123")
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    env = {
        e["name"]: e["value"] for e in yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    }
    assert env["MLFLOW_EXPERIMENT_ID"] == "exp-123"
    assert env["MLFLOW_TRACKING_URI"] == "databricks"


def test_dev_runs_offline_when_client_unavailable(tmp_path: pathlib.Path):
    # No stores + no auth: obj.client() raises, but tracing is best-effort, so dev still runs the
    # agent locally (it doesn't regress the offline path).
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


def test_dev_runs_without_traces_when_tracing_setup_fails(tmp_path: pathlib.Path, monkeypatch):
    # Tracing is best-effort locally: if provisioning raises (e.g. no mlflow), dev still runs the
    # agent, just without wiring any MLflow env.
    import yaml

    from databricks_mason.errors import AgentCliError

    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()

    def _boom(*a, **k):
        raise AgentCliError("MLflow is required")

    monkeypatch.setattr(dev_mod, "resolve_trace_experiment_id", _boom)
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    env_entries = yaml.safe_load((tmp_path / "app.yaml").read_text()).get("env") or []
    assert not any(e["name"].startswith("MLFLOW") for e in env_entries)
    assert db.call_args.args[0][:2] == ["apps", "run-local"]


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


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
def test_dev_custom_server_recommends_wiring_tools_in_agent_code(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(tmp_path, framework=framework).write()
    write_project_metadata(tmp_path, framework=framework, template=template)

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
    project = AgentProject.create(tmp_path, framework=framework)
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


@pytest.mark.parametrize(
    ("framework", "template"),
    [
        ("langgraph", "custom-agent-langgraph"),
        ("openai", "custom-agent-openai"),
    ],
)
def test_dev_surfaces_invalid_custom_server_manifest_before_starting(
    tmp_path: pathlib.Path,
    framework: str,
    template: str,
):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "agent.toml").write_text(
        f'schema_version = 1\n\n[agent]\nframework = "{framework}"\n'
        '\n[[tools]]\nid = "legacy"\nsource = { kind = "python", '
        'entrypoint = "agent.tools:legacy" }\n',
        encoding="utf-8",
    )
    write_project_metadata(tmp_path, framework=framework, template=template)
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


def test_dev_announces_durable_api_endpoint(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(
        tmp_path,
        framework="langgraph",
        durability_enabled=True,
    ).write()

    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "http://localhost:8000/api/invocations" in result.output
    assert "00000000-0000-4000-8000-000000000000" in result.output


def test_dev_standard_template_uses_runtime_api_without_durable_runtime(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    AgentProject.create(
        tmp_path,
        framework="langgraph",
        durability_enabled=False,
    ).write()
    (tmp_path / ".mason").mkdir()
    (tmp_path / ".mason" / "project.toml").write_text(
        'schema_version = 1\nframework = "langgraph"\ntemplate = "agent-langgraph"\n'
    )

    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "http://localhost:8000/api/invocations" in result.output


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


def test_dev_warns_when_stores_unbound(tmp_path: pathlib.Path):
    # `mason dev` never provisions stores (unlike deploy); it warns so the gap isn't silent.
    (tmp_path / "app.yaml").write_text("command: []\n")  # no agent.toml -> both unbound
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" in result.output
    assert "No session store bound" in result.output


def test_dev_silent_when_stores_bound(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\n'
        '\n[memory_store]\nname = "mem"\n\n[session_store]\nname = "sess"\n'
    )
    with (
        mock.patch.object(dev_mod, "_databricks"),
        mock.patch.object(dev_mod, "validate_stores"),
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" not in result.output
    assert "No session store bound" not in result.output
