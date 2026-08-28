"""Optional project additions for Mason-scaffolded agents."""

from __future__ import annotations

import pathlib
import re
from importlib import resources

import click
import yaml

from databricks_mason import render
from databricks_mason.errors import AgentCliError

_REQUIRED_PROJECT_FILES = (
    "app.yaml",
    "pyproject.toml",
    "runtime/main.py",
    "runtime/runtime.py",
)
_UI_FILES = {
    "agent/mason/durability.py": "ui_template/agent/mason/durability.py",
    "agent/mason/recovery.py": "ui_template/agent/mason/recovery.py",
    "agent/tools/long_running.py": "ui_template/agent/tools/long_running.py",
    "runtime/ui.py": "ui_template/runtime/ui.py",
    "ui/index.html": "ui_template/ui/index.html",
    "ui/styles.css": "ui_template/ui/styles.css",
    "ui/app.js": "ui_template/ui/app.js",
    "tests/test_demo_ui.py": "ui_template/tests/test_demo_ui.py",
    "tests/test_durability.py": "ui_template/tests/test_durability.py",
    "tests/test_recovery.py": "ui_template/tests/test_recovery.py",
}
_RUNTIME_IMPORT = "from runtime.runtime import build_app"
_UI_IMPORT = "from runtime.ui import install_ui"
_UI_CALL = "install_ui(app, session_history=agent.agent.session_history)"
_APP_BUILD_PATTERN = re.compile(r"^app = build_app\(.+\)$", re.MULTILINE)
_STOP_ENV = "MASON_DEMO_STOP_ENABLED"


def _validate_project(project: pathlib.Path) -> None:
    missing = [name for name in _REQUIRED_PROJECT_FILES if not (project / name).is_file()]
    if missing:
        raise AgentCliError(
            f"'{project}' is not a supported Mason agent project.",
            hint=f"Missing: {', '.join(missing)}. Run this inside a `mason init` project.",
        )


def _is_installed(main_text: str) -> bool:
    return _UI_IMPORT in main_text and "install_ui(app" in main_text


def _patched_runtime_main(main_path: pathlib.Path) -> str | None:
    text = main_path.read_text()
    if _is_installed(text):
        return None
    if _RUNTIME_IMPORT not in text:
        raise AgentCliError(
            f"Cannot add the UI to '{main_path}'.",
            hint="The runtime entry point does not match the LangGraph scratch template.",
        )
    if _APP_BUILD_PATTERN.search(text) is None:
        raise AgentCliError(
            f"Cannot add the UI to '{main_path}'.",
            hint="No module-level `app = build_app(...)` assignment was found.",
        )
    text = text.replace(_RUNTIME_IMPORT, f"{_RUNTIME_IMPORT}\n{_UI_IMPORT}", 1)
    patched_match = _APP_BUILD_PATTERN.search(text)
    assert patched_match is not None
    return f"{text[: patched_match.end()]}\n{_UI_CALL}{text[patched_match.end() :]}"


def _resource_text(resource_name: str) -> str:
    return resources.files("databricks_mason").joinpath(resource_name).read_text(encoding="utf-8")


def _install_files(project: pathlib.Path, *, overwrite: bool = False) -> list[str]:
    collisions = [name for name in _UI_FILES if (project / name).exists()]
    if collisions and not overwrite:
        raise AgentCliError(
            "Refusing to overwrite existing UI files.",
            hint=f"Existing: {', '.join(collisions)}. Move them aside or integrate manually.",
        )
    for destination, resource_name in _UI_FILES.items():
        target = project / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_resource_text(resource_name))
    return list(_UI_FILES)


def _document_stop_setting(project: pathlib.Path) -> None:
    example = project / ".env.example"
    text = example.read_text() if example.exists() else ""
    if _STOP_ENV in text:
        return
    suffix = (
        "\n# --- Optional: Mason demo stop/start durability control ---\n"
        "# Demo-only. Lets the UI stop the process so a supervisor or Databricks Apps can restart it.\n"
        f"# {_STOP_ENV}=true\n"
        "# MASON_DEMO_HEARTBEAT_SECONDS=3\n"
        "# MASON_DEMO_STALE_SECONDS=10\n"
    )
    example.write_text(f"{text.rstrip()}\n{suffix}" if text else suffix.lstrip())


def _set_dotenv(project: pathlib.Path) -> None:
    env_path = project / ".env"
    example = project / ".env.example"
    text = (
        env_path.read_text()
        if env_path.exists()
        else example.read_text()
        if example.exists()
        else ""
    )
    pattern = re.compile(rf"^{re.escape(_STOP_ENV)}=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(f"{_STOP_ENV}=true", text, count=1)
    else:
        text = f"{text.rstrip()}\n{_STOP_ENV}=true\n" if text else f"{_STOP_ENV}=true\n"
    env_path.write_text(text)


def _set_app_yaml_env(project: pathlib.Path) -> None:
    app_yaml = project / "app.yaml"
    text = app_yaml.read_text()
    document = yaml.safe_load(text) or {}
    env = document.get("env")
    if env is None:
        block = (
            "\n# Demo-only: enables the Mason UI stop/start durability control.\n"
            "env:\n"
            f"  - name: {_STOP_ENV}\n"
            '    value: "true"\n'
        )
        app_yaml.write_text(f"{text.rstrip()}\n{block}")
        return
    if not isinstance(env, list):
        raise AgentCliError("app.yaml has an unsupported `env` value; expected a list.")
    for item in env:
        if isinstance(item, dict) and item.get("name") == _STOP_ENV:
            if str(item.get("value", "")).lower() == "true":
                return
            raise AgentCliError(
                f"app.yaml already defines {_STOP_ENV} with a non-true value.",
                hint="Set it to true manually to enable the demo stop endpoint.",
            )
    lines = text.splitlines()
    env_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == "env:" and not line.startswith((" ", "\t"))
        ),
        None,
    )
    if env_start is None:
        raise AgentCliError("Could not locate the top-level `env` block in app.yaml.")
    insert_at = len(lines)
    for index in range(env_start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t", "#")):
            insert_at = index
            break
    lines[insert_at:insert_at] = [f"  - name: {_STOP_ENV}", '    value: "true"']
    app_yaml.write_text("\n".join(lines) + "\n")


@click.group(name="add")
def add() -> None:
    """Add optional capabilities to a Mason agent project."""


@add.command(name="ui")
@click.argument("directory", required=False, default=".")
@click.option(
    "--enable-stop",
    is_flag=True,
    help="Enable the demo-only endpoint that stops the app process for durability testing.",
)
@click.option(
    "--enable-crash",
    "legacy_enable_crash",
    is_flag=True,
    hidden=True,
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite Mason-managed UI files to refresh an existing installation.",
)
@click.pass_obj
def add_ui(
    obj,
    directory: str,
    enable_stop: bool,
    legacy_enable_crash: bool,
    force: bool,
) -> None:
    """Add a zero-build chat and runtime demo UI to DIRECTORY."""
    enable_stop = enable_stop or legacy_enable_crash
    project = pathlib.Path(directory).expanduser().resolve()
    _validate_project(project)

    main_path = project / "runtime/main.py"
    installed = _is_installed(main_path.read_text())
    if installed:
        missing_ui = [name for name in _UI_FILES if not (project / name).is_file()]
        if missing_ui and not force:
            raise AgentCliError(
                "The Mason UI installation is incomplete.",
                hint=(
                    f"Missing: {', '.join(missing_ui)}. Restore them, remove the UI wiring, "
                    "or rerun with --force."
                ),
            )
    patched_main = _patched_runtime_main(main_path)
    files = _install_files(project, overwrite=force) if force or not installed else []
    if patched_main is not None:
        main_path.write_text(patched_main)
    _document_stop_setting(project)
    if enable_stop:
        _set_dotenv(project)
        _set_app_yaml_env(project)

    payload = {
        "directory": str(project),
        "installed": bool(files or patched_main),
        "updated": bool(installed and force),
        "stop_enabled": enable_stop,
        "files": files,
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        return

    next_steps = [
        f"cd {project}",
        "uv run start-server",
        "Open http://localhost:8000",
    ]
    if not enable_stop:
        next_steps.append("mason add ui --enable-stop .   # opt in to the stop/start demo")
    else:
        next_steps.append("Deploy with --with-session-store to verify recovery across restarts")
    render.success(
        (
            "Updated agent demo UI"
            if installed and force
            else "Added agent demo UI"
            if files or patched_main
            else "Agent demo UI already installed"
        ),
        fields={
            "Directory": str(project),
            "Stop control": "enabled" if enable_stop else "disabled",
        },
        next_steps=next_steps,
    )
