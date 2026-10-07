"""`agentbricks experimental prompts` — declare the MLflow Prompt Registry prompts the agent loads.

Each prompt is a binding in agent.toml's ``[prompts]`` table (``key = "catalog.schema.name"``).
`agentbricks deploy` grants the app's service principal what the Prompt Registry requires to load
them, and `agentbricks experimental models upgrade` rewrites them alongside the models. This module
also holds the Prompt Registry helpers `models apply` and `models rollback` use to move aliases.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks import render

PRODUCTION_ALIAS = "production"


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Agent Bricks project containing agent.toml.",
    )(function)


def registry_mlflow(obj):
    """MLflow pointed at the workspace and its Unity Catalog prompt registry, honoring --profile."""
    import os  # noqa: PLC0415

    from databricks_agentbricks.cli import tracing  # noqa: PLC0415

    if obj.profile:
        # promote_to_prod's workspace client and MLflow both resolve auth from the environment.
        os.environ["DATABRICKS_CONFIG_PROFILE"] = obj.profile
    mlflow = tracing._mlflow()
    tracing._set_tracking_uri(mlflow, obj.profile)
    mlflow.set_registry_uri("databricks-uc")
    return mlflow


def prompt_alias_version(obj, prompt: dict) -> int:
    """Read the live alias, bypassing MLflow's local cache."""
    mlflow = registry_mlflow(obj)
    return int(
        mlflow.genai.load_prompt(
            f"prompts:/{prompt['name']}@{prompt['alias']}", cache_ttl_seconds=0
        ).version
    )


def restore_prompt_aliases(obj, prompts: list[dict]) -> None:
    """Move each prompt's alias back to the version it pointed at before an apply."""
    mlflow = registry_mlflow(obj)
    with render.status("Moving prompt aliases back…"):
        for prompt in prompts:
            mlflow.genai.set_prompt_alias(
                name=prompt["name"], alias=prompt["alias"], version=prompt["prior_version"]
            )


def _production_version(mlflow, name: str) -> Optional[str]:
    try:
        return str(mlflow.genai.load_prompt(f"prompts:/{name}@{PRODUCTION_ALIAS}").version)
    except Exception:  # noqa: BLE001 — a missing prompt or alias shows as "-", not an error
        return None


@click.group()
def prompts() -> None:
    """Declare the Prompt Registry prompts your agent loads.

    Bound prompts get the access they need at deploy, and `models upgrade` rewrites them alongside
    the models behind each LLM call.
    """


@prompts.command("bind")
@click.argument("prompt")
@click.option(
    "--key",
    default=None,
    help="Name for this prompt in agent.toml (default: the prompt's own name, e.g. writer).",
)
@_source_option
@click.pass_obj
def prompts_bind(obj, prompt: str, key: Optional[str], source: pathlib.Path) -> None:
    """Declare MLflow Prompt Registry prompt PROMPT (catalog.schema.name) that the agent loads.

    This only edits agent.toml. `agentbricks deploy` grants the app's service principal what the
    Prompt Registry requires to load prompts (USE SCHEMA, EXECUTE, CREATE FUNCTION, and MANAGE on
    the prompt's schema), and `agentbricks experimental models upgrade` optimizes bound prompts.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    key = project.bind_prompt(prompt, key)
    project.write()
    if obj.output == "json":
        render.emit_json({"key": key, "prompt": prompt, "manifest": str(project.path)})
        return
    render.success(
        f"Bound prompt '{prompt}' as '{key}'",
        fields={"agent.toml": str(project.path)},
        next_steps=[("agentbricks deploy <name>", "Grant the app access to it")],
    )


@prompts.command("unbind")
@click.argument("key")
@_source_option
@click.pass_obj
def prompts_unbind(obj, key: str, source: pathlib.Path) -> None:
    """Remove prompt binding KEY from agent.toml (the prompt itself is left in place)."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    removed = project.unbind_prompt(key)
    project.write()
    if obj.output == "json":
        render.emit_json({"key": key, "removed": removed})
        return
    render.success(f"Unbound prompt '{key}'" if removed else f"No prompt was bound as '{key}'")


@prompts.command("list")
@_source_option
@click.pass_obj
def prompts_list(obj, source: pathlib.Path) -> None:
    """List the prompts bound in agent.toml and the version each one's @production alias points at."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    mlflow = registry_mlflow(obj) if project.prompts else None
    rows = {
        key: {"prompt": name, "production_version": _production_version(mlflow, name)}
        for key, name in project.prompts.items()
    }
    if obj.output == "json":
        render.emit_json(rows)
        return
    render.resource_table(
        "Prompts",
        [("Key", "left"), ("Prompt", "left"), ("@production", "left")],
        [[key, r["prompt"], r["production_version"] or "-"] for key, r in rows.items()],
    )
