"""`mason init` — scaffold a local agent project from a mason template.

Copies one template bundled in the databricks_mason package into a local target directory, ready
for `mason deploy --source <dir>`. Because the template ships with the package, the scaffold always
matches the installed CLI; to try a fork or branch, install that mason and re-run init.

The Mason server is durable by default. Pass `--no-durable-runtime` for process-local background
state, or `--server custom` for a minimal foreground-only FastAPI server.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from importlib import resources
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as _distribution
from importlib.metadata import version as _installed_version
from typing import Optional

import click
import tomlkit

from databricks_mason import mason_source, render
from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata

# Templates ship inside this package (databricks_mason/templates/), so `mason init` always copies
# the one for the installed CLI — the scaffold can't drift from the databricks-mason it runs
# against. To try a fork/branch, install that mason (`pip install -e` or `pip install git+…@ref`);
# init then copies its bundled template and pins the SDK to that same source.

# Framework -> template name (a directory under databricks_mason/templates/).
_TEMPLATES = {"openai": "agent-openai", "langgraph": "agent-langgraph"}
_CUSTOM_SERVER_TEMPLATES = {"openai": "custom-agent-openai", "langgraph": "custom-agent-langgraph"}
_CHAT_APP_TEMPLATES = {"langgraph": "ui/agent-langgraph", "openai": "ui/agent-openai"}


def _git(args: list[str], *, cwd: Optional[pathlib.Path] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise AgentCliError(
            f"`git {' '.join(args)}` failed (exit {result.returncode})", hint=detail
        )
    return result


def _editable_checkout_root() -> pathlib.Path | None:
    """The repo root when Mason is imported from an editable checkout of databricks-ai-bridge.

    True only for a `pip install -e` from a clone of this repo: the running init.py must be the
    very file tracked in that checkout. A registry (wheel) install lives in site-packages, outside
    any repo, so this returns None and `init` uses the released tag / `main` instead. This is what
    turns on the editable dev loop — templates copied from the working tree, databricks-mason
    linked back to it — so a Mason developer's uncommitted edits show up without a commit.
    """
    module = pathlib.Path(__file__).resolve()
    try:
        root = pathlib.Path(
            (_git(["rev-parse", "--show-toplevel"], cwd=module.parent).stdout or "").strip()
        ).resolve()
    except AgentCliError:
        return None
    source_module = root / "integrations" / "mason" / "src" / "databricks_mason" / "init.py"
    if not source_module.is_file() or source_module.resolve() != module:
        return None
    return root


def _installed_git_template_source() -> tuple[str, str] | None:
    """Return the repository and commit recorded for a Git-installed Mason package."""
    try:
        direct_url = _distribution("databricks-mason").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    vcs = metadata.get("vcs_info")
    if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
        return None
    repository = metadata.get("url")
    commit = vcs.get("commit_id")
    if not isinstance(repository, str) or not isinstance(commit, str):
        return None
    return repository, commit


def _copy_packaged_template(
    name: str,
    dest: pathlib.Path,
    overlay_names: tuple[str, ...] = (),
) -> None:
    """Copy a template (and any overlays) bundled in the databricks_mason package into `dest`.

    The templates ship with the package, so a scaffold always matches the installed CLI. For an
    editable install `resources.files` resolves to the source tree, so a Mason developer's
    uncommitted template edits are scaffolded too — no git fetch, no version matching.
    """
    root = resources.files("databricks_mason").joinpath("templates")
    for index, rel in enumerate((name, *overlay_names)):
        src = root.joinpath(*rel.split("/"))
        if not src.is_dir():
            raise AgentCliError(f"Template '{rel}' is not bundled in databricks-mason.")
        shutil.copytree(
            str(src), dest, dirs_exist_ok=index > 0, ignore=shutil.ignore_patterns("__pycache__")
        )


def _bundled_template_ref() -> str:
    """A label for the packaged template's origin — the installed databricks-mason version."""
    try:
        return f"bundled (databricks-mason {_installed_version('databricks-mason')})"
    except PackageNotFoundError:
        return "bundled"


def _write_env(dest: pathlib.Path, profile: str) -> bool:
    """Seed a local `.env` from `.env.example` with DATABRICKS_CONFIG_PROFILE=<profile>.

    Returns True if a `.env` was written. Skips if `.env` already exists (never clobbers). The
    template reads DATABRICKS_CONFIG_PROFILE for local model auth, so this makes the scaffolded
    project runnable with `mason dev` without a manual `cp .env.example .env` step.
    """
    env_path = dest / ".env"
    if env_path.exists():
        return False
    example = dest / ".env.example"
    base = example.read_text() if example.exists() else ""
    lines, replaced = [], False
    for line in base.splitlines():
        if line.startswith("DATABRICKS_CONFIG_PROFILE="):
            lines.append(f"DATABRICKS_CONFIG_PROFILE={profile}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"DATABRICKS_CONFIG_PROFILE={profile}")
    env_path.write_text("\n".join(lines) + "\n")
    return True


def _pin_mason(
    dest: pathlib.Path,
    framework: str,
    source: dict,
    *,
    runtime_extra: bool = True,
) -> None:
    """Write databricks-mason's [tool.uv.sources] pin, after checking the template declares it.

    `source` is a pin table built by `mason_source` (git / editable / wheel).
    """
    pyproject = dest / "pyproject.toml"
    if not pyproject.is_file():
        return
    dependencies = tomlkit.parse(pyproject.read_text())["project"]["dependencies"]
    extra = "runtime-openai" if framework == "openai" else "runtime"
    expected_prefix = f"databricks-mason[{extra}]" if runtime_extra else "databricks-mason"
    if not any(str(dependency).startswith(expected_prefix) for dependency in dependencies):
        raise AgentCliError(
            "The selected template does not declare the expected databricks-mason dependency."
        )
    mason_source.write(pyproject, source)


def _resolve_mason_pin() -> Optional[dict]:
    """The scaffold's databricks-mason [tool.uv.sources] pin, or None to keep the template's PyPI pin.

    The template always comes from the installed package; only the SDK source varies by install:
      - editable checkout (`pip install -e`) → editable-path pin, so a Mason developer's edits show
        up in `mason dev`;
      - Git install (`pip install git+…@sha`) → git pin to the recorded commit;
      - a plain registry install → None.
    """
    checkout = _editable_checkout_root()
    if checkout is not None:
        return mason_source.editable(checkout / "integrations" / "mason")
    git_install = _installed_git_template_source()
    if git_install is not None:
        return mason_source.git(*git_install)
    return None


@click.command(name="init")
@click.argument("directory", required=False)
@click.option(
    "--framework",
    type=click.Choice(sorted(_TEMPLATES)),
    default=None,
    help="Agent framework to scaffold (defaults to langgraph).",
)
@click.option(
    "--server",
    type=click.Choice(["mason", "custom"]),
    default="mason",
    show_default=True,
    help="Use Mason's invocation server or a minimal custom FastAPI server.",
)
@click.option(
    "--no-durable-runtime",
    is_flag=True,
    hidden=True,
    help="Keep Mason server background state in-process instead of provisioning Lakebase.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed a local .env with this DATABRICKS_CONFIG_PROFILE so `mason dev` works "
    "immediately (defaults to the profile from -p / `mason login`).",
)
@click.option(
    "--disable-chat-app",
    is_flag=True,
    help="Scaffold the API-only backend, without the browser chat app.",
)
@click.option(
    "--enable-chat-app",
    is_flag=True,
    hidden=True,
    help="Deprecated: the chat app is included by default; this flag is a no-op.",
)
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    framework: Optional[str],
    server: str,
    no_durable_runtime: bool,
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
) -> None:
    """Scaffold a local agent project from a mason template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `mason dev` right away.

    The default Mason server supports foreground, streaming, and background invocations through one
    HTTP contract with a durable runtime. Pass --server custom for a minimal foreground-only
    FastAPI server.
    """
    selected_framework = framework or "langgraph"
    mason_server = server == "mason"
    if not mason_server and no_durable_runtime:
        raise click.UsageError("--no-durable-runtime only applies to --server mason")
    durable_runtime = mason_server and not no_durable_runtime
    templates = _TEMPLATES if mason_server else _CUSTOM_SERVER_TEMPLATES
    template_name = templates[selected_framework]
    chat_app_enabled = (
        mason_server and selected_framework in _CHAT_APP_TEMPLATES and not disable_chat_app
    )
    dest = pathlib.Path(directory) if directory else pathlib.Path(template_name)

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )

    overlay_names = (_CHAT_APP_TEMPLATES[selected_framework],) if chat_app_enabled else ()
    mason_pin = _resolve_mason_pin()
    try:
        # Copy the template bundled with the installed CLI, then pin databricks-mason per how that
        # CLI is installed (editable / git checkout, else the template's own PyPI pin).
        _copy_packaged_template(template_name, dest, overlay_names)
        if mason_pin is not None:
            _pin_mason(dest, selected_framework, mason_pin, runtime_extra=mason_server)
        template_ref = _bundled_template_ref()
        write_project_metadata(dest, framework=selected_framework, template=template_name)
        project = AgentProject.create(
            dest,
            framework=selected_framework,
            durability_enabled=durable_runtime,
        )
        project.write()
        env_profile = profile or obj.profile
        wrote_env = _write_env(dest, env_profile) if env_profile else False
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    if obj.output == "json":
        render.emit_json(
            {
                "framework": selected_framework,
                "template": template_name,
                "template_ref": template_ref,
                "directory": str(dest),
                "server": server,
                "chat_app_enabled": chat_app_enabled,
                "durable_runtime": durable_runtime,
                "env_profile": env_profile if wrote_env else None,
            }
        )
        return

    fields = {
        "Framework": selected_framework,
        "Server": "Mason AgentApp" if mason_server else "Custom FastAPI",
        "Template ref": template_ref,
        "Durable runtime": "enabled" if durable_runtime else "disabled",
        "Directory": str(dest),
    }
    if chat_app_enabled:
        fields["Chat app"] = "enabled"
    steps: list[str | tuple[str, str]] = [(f"cd {dest}", "Enter the project directory")]
    if wrote_env:
        fields["Profile (.env)"] = env_profile
    else:
        # No profile resolved, so no .env was seeded — call out the auth step explicitly rather
        # than burying it, since running locally fails without a Databricks profile.
        steps += [
            ("cp .env.example .env", "Create your local env file"),
            "Set DATABRICKS_CONFIG_PROFILE in .env (or re-run `mason init --profile <profile>`)",
        ]
    steps.append(("mason dev", "Run the agent locally"))
    if chat_app_enabled:
        steps.append("Open http://localhost:8000 to chat with it")
    steps.append((f"mason deploy {dest.name}", "Deploy it to Databricks (from the project dir)"))
    render.success(f"Scaffolded '{template_name}'", fields=fields, next_steps=steps)
