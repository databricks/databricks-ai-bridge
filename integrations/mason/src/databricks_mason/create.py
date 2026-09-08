"""`mason create` — interactively configure and generate a Mason agent."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Optional

import click

from databricks_mason import render
from databricks_mason.agent_config import AgentConfig
from databricks_mason.errors import AgentCliError
from databricks_mason.init import scaffold_project
from databricks_mason.project_renderer import render_project

_DEFAULT_DIRECTORY = "my-agent"
_DEFAULT_FRAMEWORK = "langgraph"
_DEFAULT_MODEL = "databricks-gpt-5-2"
_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
_FRAMEWORKS = ("langgraph", "openai")


def _prompt_value(value: str | None, label: str, default: str) -> str:
    if value is not None:
        return value
    return click.prompt(label, default=default, show_default=True, type=str)


def _collect_config(
    *,
    directory: str | None,
    name: str | None,
    framework: str | None,
    model: str | None,
    instructions: str | None,
    chat_app: bool | None,
    interactive: bool,
) -> tuple[pathlib.Path, AgentConfig]:
    if interactive:
        click.echo("Create a Mason agent")
        click.echo()
        directory = _prompt_value(directory, "Project directory", _DEFAULT_DIRECTORY)
        default_name = pathlib.Path(directory).name or _DEFAULT_DIRECTORY
        name = _prompt_value(name, "Agent name", default_name)
        if framework is None:
            framework = click.prompt(
                "Agent framework",
                default=_DEFAULT_FRAMEWORK,
                show_default=True,
                type=click.Choice(_FRAMEWORKS),
            )
        model = _prompt_value(model, "Model serving endpoint", _DEFAULT_MODEL)
        instructions = _prompt_value(
            instructions,
            "Agent instructions",
            _DEFAULT_INSTRUCTIONS,
        )
        if chat_app is None:
            chat_app = click.confirm("Include the browser chat app?", default=True)
    else:
        directory = directory or _DEFAULT_DIRECTORY
        name = name or pathlib.Path(directory).name or _DEFAULT_DIRECTORY
        framework = framework or _DEFAULT_FRAMEWORK
        model = model or _DEFAULT_MODEL
        instructions = instructions or _DEFAULT_INSTRUCTIONS
        chat_app = True if chat_app is None else chat_app

    config = AgentConfig(
        name=name,
        framework=framework,
        model=model,
        instructions=instructions,
        chat_app_enabled=bool(chat_app),
    )
    destination = pathlib.Path(directory).expanduser()

    if interactive:
        click.echo()
        click.echo(f"  Framework:  {config.framework}")
        click.echo(f"  Model:      {config.model}")
        click.echo(f"  Directory:  {destination}")
        click.echo(f"  Chat app:   {'enabled' if config.chat_app_enabled else 'disabled'}")
        click.echo()
        if not click.confirm("Create this project?", default=True):
            raise click.Abort()

    return destination, config


def create_agent_project(
    destination: pathlib.Path,
    config: AgentConfig,
    *,
    profile: str | None,
    repo: str | None,
    ref: str | None,
) -> pathlib.Path:
    """Stage, render, and atomically publish one generated agent project."""
    target = destination.resolve()
    if target.exists():
        raise AgentCliError(
            f"Destination '{destination}' already exists.",
            hint="Choose a new directory or remove the existing one.",
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}-mason-create-",
            dir=target.parent,
        ) as temporary_directory:
            staged = pathlib.Path(temporary_directory) / target.name
            scaffold_project(
                staged,
                framework=config.framework,
                profile=profile,
                chat_app_enabled=config.chat_app_enabled,
                repo=repo,
                ref=ref,
            )
            config.write(staged)
            render_project(staged, config)
            staged.replace(target)
    except AgentCliError:
        raise
    except OSError as exc:
        raise AgentCliError(f"Could not create Mason project at {destination}: {exc}.") from exc
    return target


@click.command(name="create")
@click.argument("directory", required=False)
@click.option("--name", default=None, help="Agent name stored in agent.toml.")
@click.option(
    "--framework",
    type=click.Choice(_FRAMEWORKS),
    default=None,
    help=f"Agent framework (default: {_DEFAULT_FRAMEWORK}).",
)
@click.option(
    "--model",
    default=None,
    help=f"Databricks model serving endpoint (default: {_DEFAULT_MODEL}).",
)
@click.option(
    "--instructions",
    default=None,
    help="System instructions rendered into the generated agent code.",
)
@click.option(
    "--chat-app/--no-chat-app",
    default=None,
    help="Include the browser chat app (default: enabled).",
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    show_default=True,
    help="Prompt for missing configuration and confirm before writing files.",
)
@click.option(
    "--profile",
    default=None,
    help="Seed .env with this DATABRICKS_CONFIG_PROFILE.",
)
@click.option("--repo", default=None, help="Override the git repo URL for the base template.")
@click.option("--ref", default=None, help="Override the branch, tag, or ref for the base template.")
@click.pass_obj
def create(
    obj,
    directory: Optional[str],
    name: Optional[str],
    framework: Optional[str],
    model: Optional[str],
    instructions: Optional[str],
    chat_app: Optional[bool],
    interactive: bool,
    profile: Optional[str],
    repo: Optional[str],
    ref: Optional[str],
) -> None:
    """Walk through configuration and generate a runnable agent project.

    With no flags, Mason prompts for the project directory, agent name, framework, model endpoint,
    instructions, and browser chat app. Use --no-interactive for scripts; omitted values then use
    the same defaults shown by the wizard.
    """
    if interactive and obj.output == "json":
        raise AgentCliError(
            "Interactive create cannot be combined with --output json.",
            hint="Pass --no-interactive and provide any values you want to override.",
        )

    destination, config = _collect_config(
        directory=directory,
        name=name,
        framework=framework,
        model=model,
        instructions=instructions,
        chat_app=chat_app,
        interactive=interactive,
    )
    create_agent_project(
        destination,
        config,
        profile=profile or obj.profile,
        repo=repo,
        ref=ref,
    )

    if obj.output == "json":
        render.emit_json(
            {
                "name": config.name,
                "framework": config.framework,
                "model": config.model,
                "instructions": config.instructions,
                "directory": str(destination),
                "chat_app_enabled": config.chat_app_enabled,
                "profile": profile or obj.profile,
            }
        )
        return

    fields = {
        "Agent": config.name,
        "Framework": config.framework,
        "Model": config.model,
        "Directory": str(destination),
        "Chat app": "enabled" if config.chat_app_enabled else "disabled",
    }
    next_steps: list[str | tuple[str, str]] = [
        (f"cd {destination}", "Enter the project directory"),
        ("mason dev", "Run the agent locally"),
    ]
    if config.chat_app_enabled:
        next_steps.append("Open http://localhost:8000 to chat with it")
    next_steps.append((f"mason deploy {config.name}", "Deploy it to Databricks Apps"))
    render.success("Mason agent created", fields=fields, next_steps=next_steps)
