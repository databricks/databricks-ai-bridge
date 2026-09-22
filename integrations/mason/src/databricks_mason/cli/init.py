"""`mason init` — scaffold a local agent project from a mason template.

Copies one template bundled in the databricks_mason package into a local target directory, ready
for `mason deploy --source <dir>`. Because the template ships with the package, the scaffold always
matches the installed CLI; to try a fork or branch, install that mason and re-run init.

The Mason server provisions its Runtime Store at deployment. Use `--server custom` for a minimal
foreground-only FastAPI server.
"""

from __future__ import annotations

import json
import pathlib
import secrets
import shlex
import shutil
import string
import tempfile
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, default_store_name
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata
from databricks_mason.project_types import (
    AgentFramework,
    AgentServer,
    parse_framework,
    parse_server,
)

# Templates ship inside this package (databricks_mason/templates/), so `mason init` always copies
# the one for the installed CLI — the scaffold can't drift from the databricks-mason it runs
# against. For an editable install `resources.files` resolves to the source tree, so a Mason
# developer's uncommitted template edits are scaffolded too.


@dataclass(frozen=True)
class _AgentTemplate:
    """The bundled templates for one framework (directories under databricks_mason/templates/)."""

    mason_server: str  # scaffold for Mason's invocation server (the default)
    custom_server: str  # scaffold for the minimal custom FastAPI server (--server custom)
    chat_app: str  # browser chat-app template, overlaid on the Mason-server scaffold


# Framework -> its bundled templates.
_TEMPLATES = {
    AgentFramework.LANGGRAPH: _AgentTemplate(
        "agent-langgraph", "custom-agent-langgraph", "ui/agent-langgraph"
    ),
    AgentFramework.OPENAI: _AgentTemplate("agent-openai", "custom-agent-openai", "ui/agent-openai"),
}

# Framework -> display label for user-facing text (prompts, help).
_FRAMEWORK_LABELS = {
    AgentFramework.LANGGRAPH: "LangGraph",
    AgentFramework.OPENAI: "OpenAI Agents SDK",
}


def _or_join(items: list[str]) -> str:
    """Join items into a human-readable phrase: "a", "a or b", or "a, b, or c"."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


# Derived from _FRAMEWORK_LABELS / AgentFramework so user-facing text never drifts from the
# supported-framework list. NOTE: these must stay above the click decorators below — Click
# evaluates `help=` strings at import time, not at call time.
_FRAMEWORK_LABEL_PHRASE = _or_join(list(_FRAMEWORK_LABELS.values()))
_FRAMEWORK_VALUE_PHRASE = ", ".join(framework.value for framework in AgentFramework)

_MIGRATION_DIR = "mason-migrate"
# Each coding agent discovers skills in its own configuration directory, so the bundle lives in one
# tool-neutral directory and every agent gets a pointer to it rather than a copy of the reference.
_POINTER_ROOTS = (".claude", ".agent")


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


def _migration_paths(dest: pathlib.Path, target: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """`target` plus every directory under `dest` that Mason would have to create to reach it."""
    parts = target.relative_to(dest).parts
    return tuple(dest.joinpath(*parts[:index]) for index in range(1, len(parts) + 1))


def _pointer_skill(skill: pathlib.Path) -> str:
    """A skill file forwarding an agent to the one bundle, so the reference is never duplicated."""
    _, frontmatter, _ = skill.read_text(encoding="utf-8").split("---", 2)
    target = f"../../../{_MIGRATION_DIR}/SKILL.md"
    return (
        f"---{frontmatter}---\n\n"
        f"The migration bundle lives at [{target}]({target}), outside any single agent's "
        "configuration directory.\n\nRead that file and follow it. Its `references/` directory "
        "holds the migration settings and the candidate project generated from the templates "
        "bundled with the installed CLI.\n"
    )


def _install_migration(
    staged: pathlib.Path, bundle: pathlib.Path, pointers: tuple[pathlib.Path, ...]
) -> None:
    """Install the staged bundle, removing anything created here if a later write fails."""
    pointer_skill = _pointer_skill(staged / "SKILL.md")
    created: list[pathlib.Path] = []
    try:
        shutil.copytree(staged, bundle)
        created.append(bundle)
        for pointer in pointers:
            # Remember the outermost directory Mason creates so cleanup never removes a
            # pre-existing agent configuration directory.
            outermost = pointer
            while not outermost.parent.exists():
                outermost = outermost.parent
            pointer.mkdir(parents=True)
            created.append(outermost)
            (pointer / "SKILL.md").write_text(pointer_skill, encoding="utf-8")
    except Exception:
        for path in created:
            shutil.rmtree(path, ignore_errors=True)
        raise


def _prepare_migration(
    obj,
    dest: pathlib.Path,
    *,
    framework: AgentFramework,
    chat_app_enabled: bool,
    profile: Optional[str],
    memory_store: Optional[str],
    session_store: Optional[str],
) -> None:
    """Prepare a reference project and migration instructions without changing the application."""
    if not dest.is_dir():
        raise AgentCliError(f"Existing project directory '{dest}' was not found.")
    bundle = dest / _MIGRATION_DIR
    pointers = tuple(dest / root / "skills" / _MIGRATION_DIR for root in _POINTER_ROOTS)
    for target in (bundle, *pointers):
        for path in _migration_paths(dest, target):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise AgentCliError(f"Cannot write migration files at '{path}'.")
        if target.exists():
            raise AgentCliError(
                f"Migration files already exist at '{target}'.",
                hint=f"Use the existing {_MIGRATION_DIR}/PROMPT.md, or move that directory "
                "before regenerating.",
            )

    # Build the bundle before touching the project, so a failed copy leaves no partial skill.
    with tempfile.TemporaryDirectory(prefix="mason-migrate-") as tmp:
        staged = pathlib.Path(tmp) / _MIGRATION_DIR
        reference = staged / "references" / "template"
        template = _TEMPLATES[framework]
        overlays = (template.chat_app,) if chat_app_enabled else ()
        _copy_packaged_template(template.mason_server, reference, overlays)
        project_name = dest.resolve().name
        token = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))
        AgentProject.create(
            reference,
            framework=framework,
            server=AgentServer.MASON,
            memory_store=memory_store or default_store_name(project_name, "memory", token),
            session_store=session_store or default_store_name(project_name, "sessions", token),
        ).write()
        write_project_metadata(reference, framework=framework, template=template.mason_server)
        source = resources.files("databricks_mason").joinpath("templates").joinpath(_MIGRATION_DIR)
        (staged / "SKILL.md").write_text(
            source.joinpath("SKILL.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        template_ref = _bundled_template_ref()
        (staged / "references" / "migration.json").write_text(
            json.dumps(
                {
                    "framework": framework.value,
                    "template_ref": template_ref,
                    "server": "mason",
                    "chat_app_enabled": chat_app_enabled,
                    "profile": profile,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        prompt = (
            f"Use the mason-migrate skill at {_MIGRATION_DIR}/SKILL.md to adapt "
            f"my existing {_FRAMEWORK_LABELS[framework]} agent in this project for Mason. Read "
            "its migration settings and local template reference. Implement and verify the "
            "integration while preserving my agent's behavior. Explicitly handle existing "
            "persistence, conversation history, custom state, and client contracts; surface any "
            "unresolved migration choices. Report which Mason commands are ready and any "
            "remaining limitations.\n"
        )
        (staged / "PROMPT.md").write_text(prompt, encoding="utf-8")
        _install_migration(staged, bundle, pointers)

    if obj.output == "json":
        render.emit_json(
            {
                "mode": "existing",
                "framework": framework.value,
                "directory": str(dest),
                "bundle": str(bundle),
                "skill": str(bundle / "SKILL.md"),
                "prompt_file": str(bundle / "PROMPT.md"),
                "prompt": prompt.strip(),
                "pointers": [str(pointer / "SKILL.md") for pointer in pointers],
                "template_ref": template_ref,
                "chat_app_enabled": chat_app_enabled,
            }
        )
        return
    render.success(
        "Prepared migration instructions (agent conversion is still required)",
        fields={"Directory": str(dest), "Skill": str(bundle / "SKILL.md")},
        next_steps=[
            (f"cd {shlex.quote(str(dest))}", "Enter the existing project"),
            f"Open your coding agent and paste the prompt from {_MIGRATION_DIR}/PROMPT.md:",
            prompt.strip(),
        ],
    )


@click.command(name="init")
@click.argument("directory", required=False)
@click.option(
    "--existing",
    is_flag=True,
    help=f"Prepare a coding-agent migration bundle for an existing {_FRAMEWORK_LABEL_PHRASE} "
    "project (defaults to .).",
)
@click.option(
    "--framework",
    type=click.Choice([framework.value for framework in AgentFramework]),
    default=None,
    help="Agent framework to scaffold (defaults to langgraph).",
)
@click.option(
    "--server",
    type=click.Choice([server.value for server in AgentServer]),
    default=AgentServer.MASON.value,
    show_default=True,
    help="Use Mason's invocation server or a minimal custom FastAPI server.",
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
@click.option(
    "--memory-store",
    "memory_store",
    default=None,
    help="Name for the declared memory store (default: a unique name derived from the directory, "
    "<dir>-<token>-memory). Only --server mason declares stores by default.",
)
@click.option(
    "--session-store",
    "session_store",
    default=None,
    help="Name for the declared session store (default: a unique name derived from the directory, "
    "<dir>-<token>-sessions).",
)
@click.pass_obj
def init(
    obj,
    directory: Optional[str],
    existing: bool,
    framework: Optional[str],
    server: str,
    profile: Optional[str],
    disable_chat_app: bool,
    enable_chat_app: bool,
    memory_store: Optional[str],
    session_store: Optional[str],
) -> None:
    """Scaffold a local agent project from a mason template.

    DIRECTORY is the target path to create (defaults to the template's own name). The
    directory must not already exist unless --existing is supplied. Once scaffolded, deploy it with
    `mason deploy <name> --source <directory>`.

    Pass --profile (or set a default via `mason login` / -p) to seed a local `.env` so the
    scaffolded project runs with `mason dev` right away.

    The scaffold is preconfigured to call Databricks model serving through the AI Gateway using
    that profile, so it can talk to a model with no separate endpoint or API key to set up.

    The default Mason server supports foreground, streaming, and background invocations through one
    HTTP contract and Runtime Store. Pass --server custom for a minimal foreground-only
    FastAPI server.

    With --existing, prepare a skill, prompt, and bundled template reference under
    mason-migrate/, and point each supported coding agent's skills directory at it. Run the
    prompt in your coding agent to migrate the agent onto Mason; init leaves existing
    application source, dependencies, and configuration intact.
    """
    selected_framework = parse_framework(framework or AgentFramework.LANGGRAPH)
    selected_server = parse_server(server)
    mason_server = selected_server == AgentServer.MASON
    template = _TEMPLATES[selected_framework]
    template_name = template.mason_server if mason_server else template.custom_server
    chat_app_enabled = mason_server and not disable_chat_app
    if existing:
        if not mason_server:
            raise click.UsageError(
                "--existing requires --server mason (supported frameworks: "
                f"{_FRAMEWORK_VALUE_PHRASE})."
            )
        _prepare_migration(
            obj,
            pathlib.Path(directory or "."),
            framework=selected_framework,
            chat_app_enabled=chat_app_enabled,
            profile=profile or obj.profile,
            memory_store=memory_store,
            session_store=session_store,
        )
        return
    dest = pathlib.Path(directory) if directory else pathlib.Path(template_name)

    if dest.exists():
        raise AgentCliError(
            f"Destination '{dest}' already exists.",
            hint=f"Use --existing to prepare a migration from an existing "
            f"{_FRAMEWORK_LABEL_PHRASE} project, or choose a new directory to scaffold.",
        )

    overlay_names = (template.chat_app,) if chat_app_enabled else ()
    try:
        # Copy the template bundled with the installed CLI. The scaffold keeps the template's own
        # databricks-mason PyPI dependency; it can't drift from the CLI because both ship together.
        _copy_packaged_template(template_name, dest, overlay_names)
        template_ref = _bundled_template_ref()
        write_project_metadata(dest, framework=selected_framework, template=template_name)
        if mason_server:
            # Store display names aren't unique and projects often share a directory name, so a
            # per-scaffold token keeps default stores from colliding. Memory and session share the
            # token; an explicit --memory/session-store wins.
            token = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))
            memory_store = memory_store or default_store_name(dest.name, "memory", token)
            session_store = session_store or default_store_name(dest.name, "sessions", token)
        project = AgentProject.create(
            dest,
            framework=selected_framework,
            server=selected_server,
            memory_store=memory_store,
            session_store=session_store,
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
                "framework": selected_framework.value,
                "template": template_name,
                "template_ref": template_ref,
                "directory": str(dest),
                "server": selected_server.value,
                "chat_app_enabled": chat_app_enabled,
                "env_profile": env_profile if wrote_env else None,
                "memory_store": memory_store,
                "session_store": session_store,
            }
        )
        return

    fields = {
        "Framework": selected_framework.value,
        "Server": "Mason AgentApp" if mason_server else "Custom FastAPI",
        "Template ref": template_ref,
        "Directory": str(dest),
    }
    if chat_app_enabled:
        fields["Chat app"] = "enabled"
    # Surface the store names: with the random token they can't be inferred from the directory.
    if memory_store:
        fields["Memory store"] = memory_store
    if session_store:
        fields["Session store"] = session_store
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
