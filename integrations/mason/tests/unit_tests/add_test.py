import json
import pathlib

from click.testing import CliRunner

from databricks_mason.add import add


class _Ctx:
    def __init__(self, output="text"):
        self.output = output


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "agent"
    (project / "runtime").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "runtime/main.py").write_text(
        "import agent.agent\n"
        "from runtime.runtime import build_app\n\n"
        "app = build_app(agent.agent.invoke_handler, agent.agent.stream_handler)\n"
    )
    (project / "runtime/runtime.py").write_text("def build_app(*args): ...\n")
    (project / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (project / "app.yaml").write_text('command: ["uv", "run", "start-server"]\n')
    (project / ".env.example").write_text("DATABRICKS_CONFIG_PROFILE=DEFAULT\n")
    return project


def test_add_ui_installs_files_and_patches_runtime(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    result = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    assert (project / "ui/index.html").is_file()
    assert (project / "runtime/ui.py").is_file()
    assert (project / "tests/test_demo_ui.py").is_file()
    assert (project / "tests/test_durability.py").is_file()
    assert (project / "tests/test_recovery.py").is_file()
    assert (project / "agent/mason/durability.py").is_file()
    assert (project / "agent/mason/recovery.py").is_file()
    assert (project / "agent/tools/long_running.py").is_file()
    ui_script = (project / "ui/app.js").read_text()
    assert "/api/demo/memory/search" in ui_script
    assert "/api/demo/sessions/" in ui_script
    assert 'credentials: "same-origin"' in ui_script
    assert "renderSessionTranscript(items)" in ui_script
    assert "refreshSession({ hydrateChat: true })" in ui_script
    assert "/api/demo/recovery/" in ui_script
    assert "async function startApp" in ui_script
    assert "/api/demo/app/stop" in ui_script
    assert "/api/demo/app/${encodeURIComponent(sessionId)}/start" in ui_script
    assert "Approve paused HITL" in (project / "ui/index.html").read_text()
    assert "tool_step_1" in (project / "agent/tools/long_running.py").read_text()
    main = (project / "runtime/main.py").read_text()
    assert "from runtime.ui import install_ui" in main
    assert "install_ui(app, session_history=agent.agent.session_history)" in main
    assert "MASON_DEMO_STOP_ENABLED" in (project / ".env.example").read_text()


def test_add_ui_enables_stop_locally_and_when_deployed(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    result = CliRunner().invoke(add, ["ui", "--enable-stop", str(project)], obj=_Ctx())
    assert result.exit_code == 0, result.output
    env = (project / ".env").read_text()
    assert "DATABRICKS_CONFIG_PROFILE=DEFAULT" in env
    assert "MASON_DEMO_STOP_ENABLED=true" in env
    app_yaml = (project / "app.yaml").read_text()
    assert "name: MASON_DEMO_STOP_ENABLED" in app_yaml
    assert 'value: "true"' in app_yaml


def test_add_ui_can_enable_stop_after_install(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    first = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    second = CliRunner().invoke(add, ["ui", "--enable-stop", str(project)], obj=_Ctx())
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already installed" in second.output
    assert "MASON_DEMO_STOP_ENABLED=true" in (project / ".env").read_text()


def test_add_ui_keeps_enable_crash_as_hidden_compatibility_alias(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    result = CliRunner().invoke(add, ["ui", "--enable-crash", str(project)], obj=_Ctx())

    assert result.exit_code == 0, result.output
    assert "MASON_DEMO_STOP_ENABLED=true" in (project / ".env").read_text()


def test_add_ui_force_refreshes_existing_install(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    first = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    assert first.exit_code == 0, first.output
    (project / "ui/app.js").write_text("old UI")
    (project / "agent/mason/recovery.py").unlink()

    refreshed = CliRunner().invoke(add, ["ui", "--force", str(project)], obj=_Ctx())

    assert refreshed.exit_code == 0, refreshed.output
    assert "Updated agent demo UI" in refreshed.output
    assert "async function startApp" in (project / "ui/app.js").read_text()
    assert (project / "agent/mason/durability.py").is_file()
    assert (project / "agent/mason/recovery.py").is_file()


def test_add_ui_reports_incomplete_installation(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    first = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    assert first.exit_code == 0, first.output
    (project / "ui/app.js").unlink()
    second = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    assert second.exit_code != 0
    assert "installation is incomplete" in second.output
    assert "ui/app.js" in second.output
    assert "--force" in second.output


def test_add_ui_refuses_existing_unmanaged_files(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    (project / "ui").mkdir()
    (project / "ui/index.html").write_text("custom")
    result = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx())
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output
    assert (project / "ui/index.html").read_text() == "custom"


def test_add_ui_rejects_non_mason_project(tmp_path: pathlib.Path):
    result = CliRunner().invoke(add, ["ui", str(tmp_path)], obj=_Ctx())
    assert result.exit_code != 0
    assert "not a supported Mason agent project" in result.output


def test_add_ui_json_output(tmp_path: pathlib.Path):
    project = _project(tmp_path)
    result = CliRunner().invoke(add, ["ui", str(project)], obj=_Ctx(output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["directory"] == str(project.resolve())
    assert payload["installed"] is True
    assert payload["stop_enabled"] is False
