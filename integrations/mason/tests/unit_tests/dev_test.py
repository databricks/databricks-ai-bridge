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
                    {"name": "AGENT_SESSION_STORE", "value": "s"},
                    {"name": "PIP_INDEX_URL", "value": "https://pypi.org/simple/"},
                    {"name": "UV_INDEX_URL", "value": "https://pypi.org/simple/"},
                ],
            }
        )
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    names = {e["name"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert names == {"AGENT_SESSION_STORE", "DATABRICKS_MASON_RUNTIME_STORE_LOCAL"}


def test_dev_uses_local_entry_point_without_index_override(tmp_path: pathlib.Path):
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump({"command": ["x"], "env": [{"name": "AGENT_SESSION_STORE", "value": "s"}]})
    )
    dev_yaml = dev_mod._dev_entry_point(tmp_path / "app.yaml")
    env = {e["name"]: e["value"] for e in yaml.safe_load(dev_yaml.read_text())["env"]}
    assert env == {
        "AGENT_SESSION_STORE": "s",
        "DATABRICKS_MASON_RUNTIME_STORE_LOCAL": "true",
    }
    original_env = yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    assert original_env == [{"name": "AGENT_SESSION_STORE", "value": "s"}]


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


def test_dev_checks_stores_when_bound_and_keeps_app_yaml_clean(tmp_path: pathlib.Path, monkeypatch):
    # When stores are declared and exist, dev resolves them into the dev-only manifest and does NOT
    # touch the deployable app.yaml (deploy owns that; dev's overrides live in app.masondev.yaml).
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(tmp_path, memory="m", session="s")
    (tmp_path / ".venv").mkdir()
    resolve_calls: list[str] = []
    monkeypatch.setattr(
        dev_mod,
        "_resolve_memory_store",
        lambda client, name: (resolve_calls.append(name), {"name": "memory-stores/m-id"})[1],
    )
    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert resolve_calls == ["m"]  # resolved by display name
    # Store env goes into the dev-only manifest, so the deployable app.yaml stays clean.
    env_entries = yaml.safe_load((tmp_path / "app.yaml").read_text()).get("env") or []
    assert {e["name"] for e in env_entries} == set()
    assert db.call_args.args[0][:2] == ["apps", "run-local"]


def test_dev_skips_store_check_when_no_bindings(tmp_path: pathlib.Path, monkeypatch):
    # No agent.toml store bindings -> store-check block is skipped entirely.
    (tmp_path / "app.yaml").write_text("command: []\n")
    (tmp_path / ".venv").mkdir()
    resolve_calls: list[str] = []
    monkeypatch.setattr(
        dev_mod, "_resolve_memory_store", lambda client, name: resolve_calls.append(name)
    )
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert resolve_calls == []  # no store bound -> no resolution attempted


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


def test_dev_shows_default_experiment_url_when_tracing_on(tmp_path: pathlib.Path, monkeypatch):
    # Tracing is on by default: dev surfaces the default experiment's Traces URL (the same link
    # `mason deploy` prints) so a dev run makes clear where its traces land.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(dev_mod, "resolve_trace_experiment_id", lambda *a, **k: "exp-123")
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Traces" in output
    assert "/ml/experiments/exp-123" in output


def test_dev_omits_experiment_url_when_tracing_off(tmp_path: pathlib.Path, monkeypatch):
    # Tracing disabled (resolve returns None) -> no Traces line in the startup panel.
    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(dev_mod, "resolve_trace_experiment_id", lambda *a, **k: None)
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "/ml/experiments/" not in result.output


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
    # Tracing is best-effort locally: if provisioning raises (e.g. offline / no workspace access),
    # dev still runs the agent, just without wiring any MLflow env.
    import yaml

    from databricks_mason.errors import AgentCliError

    (tmp_path / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    (tmp_path / ".venv").mkdir()

    def _boom(*a, **k):
        raise AgentCliError("could not reach the workspace")

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


def test_dev_silent_when_stores_bound(tmp_path: pathlib.Path, monkeypatch):
    (tmp_path / "app.yaml").write_text("command: []\n")
    _write_agent_manifest(tmp_path, memory="mem", session="sess")
    monkeypatch.setattr(
        dev_mod,
        "_resolve_memory_store",
        lambda client, name: {"name": "memory-stores/mem-id"},
    )
    with mock.patch.object(dev_mod, "_databricks"):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(tmp_path)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert "No memory store bound" not in result.output
    assert "No session store bound" not in result.output


def test_dev_warns_when_declared_store_is_missing(tmp_path: pathlib.Path, monkeypatch):
    # init declares a store before it exists remotely; dev must warn and keep running, not error.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, memory="declared-mem")

    monkeypatch.setattr(dev_mod, "_resolve_memory_store", lambda client, name: None)  # missing

    with mock.patch.object(
        dev_mod, "_databricks", return_value=types.SimpleNamespace(returncode=0)
    ):
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(src)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "declared-mem" in result.output and "not created" in result.output.lower()


def test_dev_degrades_gracefully_when_store_client_raises_offline(
    tmp_path: pathlib.Path, monkeypatch
):
    # When a store is declared but the client RAISES (offline / no auth), dev must exit 0 with a
    # warning and must NOT abort — mirrors the tracing block's best-effort pattern.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"]}))
    _write_agent_manifest(src, memory="my-memory", session="my-session")
    (src / ".venv").mkdir()

    def _boom_resolve(client, name):
        raise RuntimeError("offline: could not reach the workspace")

    monkeypatch.setattr(dev_mod, "_resolve_memory_store", _boom_resolve)

    class _OfflineCtx:
        output = "text"
        profile = None

        def client(self):
            raise AgentCliError("no databricks auth configured")

    with mock.patch.object(dev_mod, "_databricks") as db:
        result = CliRunner().invoke(dev_mod.dev, ["--source", str(src)], obj=_OfflineCtx())

    assert result.exit_code == 0, result.output
    assert "not created" in result.output.lower()  # warning, not abort
    assert db.call_args.args[0][:2] == ["apps", "run-local"]  # agent still ran
    # No AGENT_MEMORY_STORE injected when client/resolve failed
    dev_yaml_path = src / "app.masondev.yaml"
    assert not dev_yaml_path.exists()  # cleaned up by finally block after run


def test_dev_injects_store_env_into_dev_manifest_only(tmp_path: pathlib.Path, monkeypatch):
    # When the stores exist, the memory id and session name are injected into the dev-only manifest
    # (app.masondev.yaml) but NOT into the deployable app.yaml — the runtime reads env, not agent.toml.
    src = tmp_path / "app"
    src.mkdir()
    (src / "app.yaml").write_text(yaml.safe_dump({"command": ["x"], "env": []}))
    _write_agent_manifest(src, memory="mem", session="sess")

    monkeypatch.setattr(
        dev_mod, "_resolve_memory_store", lambda client, name: {"name": "memory-stores/mem-id-123"}
    )

    captured_dev: dict = {}

    def _fake_databricks(args, *a, **kw):
        # Read the dev manifest while it still exists (before finally-block cleanup).
        dev_yaml = pathlib.Path(kw["cwd"]) / "app.masondev.yaml"
        captured_dev.update(yaml.safe_load(dev_yaml.read_text()))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dev_mod, "_databricks", _fake_databricks)

    result = CliRunner().invoke(dev_mod.dev, ["--source", str(src)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    dev_env = {e["name"]: e["value"] for e in captured_dev.get("env", [])}
    assert dev_env["AGENT_MEMORY_STORE"] == "mem-id-123"
    assert dev_env["AGENT_SESSION_STORE"] == "sess"
    real_env = {
        e["name"]: e["value"]
        for e in (yaml.safe_load((src / "app.yaml").read_text()).get("env") or [])
    }
    assert "AGENT_MEMORY_STORE" not in real_env  # real app.yaml is untouched for stores
    assert "AGENT_SESSION_STORE" not in real_env
