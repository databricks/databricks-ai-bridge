"""`agentbricks experimental models` — bind the agent's LLM calls to swappable model services.

Each LLM call site in the agent (a *role*, e.g. ``router`` and ``writer`` in a compound agent, or
the single ``agent`` role in a one-model agent) calls a user-owned Unity Catalog AI Gateway *model
service* (``catalog.schema.name``) declared in agent.toml's ``[model_services.<role>]`` table,
instead of a hardcoded ``system.ai.*`` model. `agentbricks deploy` creates each service (routed to
its binding's default model) and grants the app access; from then on the model behind a service
can be switched with ``models set``, with no code change or redeploy.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.runtime.model_services import destination_model

_BIND_COMMAND = "agentbricks experimental models bind <catalog>.<schema>.<name> [--role <role>] --default system.ai.<model>"


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Agent Bricks project containing agent.toml.",
    )(function)


def _yes_option(function):
    return click.option(
        "--yes", "-y", is_flag=True, help="Switch without asking for confirmation."
    )(function)


def _role_option(function):
    return click.option(
        "--role",
        default=None,
        help="Which bound LLM call site (e.g. router). Optional when only one role is bound.",
    )(function)


def _bound_project(source: pathlib.Path):
    """The project at ``source``; errors when it has no model-service binding."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    if not project.model_services:
        raise AgentCliError(
            "This agent has no model service bound.",
            hint=f"Run `{_BIND_COMMAND}`, then `agentbricks deploy`.",
        )
    return project


def _pick_role(project, role: Optional[str]) -> str:
    """``role`` if it's bound; the only bound role when ``role`` is omitted; otherwise an error."""
    roles = list(project.model_services)
    if role is not None:
        if role not in project.model_services:
            raise AgentCliError(
                f"No model service is bound to role '{role}'.",
                hint=f"Bound roles: {', '.join(roles)}.",
            )
        return role
    if len(roles) == 1:
        return roles[0]
    raise AgentCliError(
        "This agent binds several model services; say which one.",
        hint=f"Pass --role with one of: {', '.join(roles)}.",
    )


def _current_model(client, service: str) -> str:
    try:
        resolved = client.get_model_service(service)
    except AgentCliError as exc:
        if exc.error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
            raise AgentCliError(
                f"Model service '{service}' doesn't exist yet.",
                hint="Run `agentbricks deploy` to create it from agent.toml.",
            ) from exc
        raise
    model = destination_model(resolved)
    if model is None:
        raise AgentCliError(f"Model service '{service}' has no foundation-model destination.")
    return model


def _confirm(obj, yes: bool, question: str) -> bool:
    """True to go ahead. JSON output never prompts: it goes ahead only with --yes."""
    if yes:
        return True
    if obj.output == "json":
        return False
    return click.confirm(question, default=False)


@click.group()
def models() -> None:
    """Choose the models behind your agent.

    Bind each of the agent's LLM calls to a model service it calls through the AI Gateway, then
    switch the model behind a service at any time. Switches take effect without a code change or
    redeploy.
    """


@models.command("bind")
@click.argument("service")
@click.option(
    "--role",
    default=None,
    help="Name for the LLM call site this service backs, e.g. router or writer (default: agent). "
    "Bind one service per call site in a compound agent.",
)
@click.option(
    "--default",
    "default_model",
    default=None,
    help="system.ai.* model `agentbricks deploy` routes the service to when it creates it.",
)
@_source_option
@click.pass_obj
def models_bind(
    obj, service: str, role: Optional[str], default_model: Optional[str], source: pathlib.Path
) -> None:
    """Bind model service SERVICE (catalog.schema.name) to one of the agent's LLM calls.

    This only edits agent.toml — it does not create the service. `agentbricks deploy` creates it if it
    doesn't exist (routed to --default), grants the app's service principal EXECUTE on it, and points
    the agent at it via AGENT_MODEL_SERVICE_<ROLE>.
    """
    from databricks_agentbricks.agent_project import (  # noqa: PLC0415
        DEFAULT_MODEL_ROLE,
        AgentProject,
    )
    from databricks_agentkit.runtime.model_services import system_ai_name  # noqa: PLC0415
    from databricks_agentkit.runtime.tool_manifest import model_service_env  # noqa: PLC0415

    role = role or DEFAULT_MODEL_ROLE
    project = AgentProject.load(source)
    project.bind_model_service(
        service, system_ai_name(default_model) if default_model else None, role=role
    )
    project.write()
    binding = project.model_services[role]
    if obj.output == "json":
        render.emit_json(
            {
                "role": role,
                "model_service": service,
                "default": binding.default,
                "manifest": str(project.path),
            }
        )
        return
    fields = {"agent.toml": str(project.path), "Env var": model_service_env(role)}
    if binding.default:
        fields["Default model"] = binding.default
    render.success(
        f"Bound model service '{service}' to role '{role}'",
        fields=fields,
        next_steps=[
            ("agentbricks deploy <name>", "Create it if missing and grant the app access"),
            ("agentbricks experimental models set <model>", "Then switch it to another model"),
        ],
    )


@models.command("unbind")
@_role_option
@_source_option
@click.pass_obj
def models_unbind(obj, role: Optional[str], source: pathlib.Path) -> None:
    """Remove a model-service binding from agent.toml.

    The service itself is left in place. After the next deploy that LLM call uses its own default
    model directly again.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    removed = False
    if project.model_services:
        role = _pick_role(project, role)
        removed = project.unbind_model_service(role)
        project.write()
    if obj.output == "json":
        render.emit_json({"role": role, "removed": removed})
        return
    render.success(f"Unbound role '{role}'" if removed else "No model service was bound")


@models.command("list")
@click.pass_obj
def models_list(obj) -> None:
    """List the system.ai.* chat models you can route the agent to."""
    names = obj.client().list_chat_model_services()
    if obj.output == "json":
        render.emit_json(names)
        return
    render.resource_table(
        "AI Gateway Models · system.ai", [("Model", "left")], [[n] for n in names]
    )


@models.command("status")
@_source_option
@click.pass_obj
def models_status(obj, source: pathlib.Path) -> None:
    """Show each bound model service and the model behind it now."""
    project = _bound_project(source)
    client = obj.client()
    bound = {
        role: {"model_service": b.name, "model": _current_model(client, b.name)}
        for role, b in project.model_services.items()
    }
    if obj.output == "json":
        render.emit_json({"model_services": bound})
        return
    render.resource_table(
        "Model services",
        [("Role", "left"), ("Model service", "left"), ("Current model", "left")],
        [[role, b["model_service"], b["model"]] for role, b in bound.items()],
    )


@models.command("set")
@click.argument("model")
@_role_option
@_yes_option
@_source_option
@click.pass_obj
def models_set(obj, model: str, role: Optional[str], yes: bool, source: pathlib.Path) -> None:
    """Switch one bound model service to MODEL (a system.ai.* name)."""
    from databricks_agentkit.runtime.model_services import system_ai_name  # noqa: PLC0415

    project = _bound_project(source)
    service = project.model_services[_pick_role(project, role)].name
    model = system_ai_name(model)
    previous = _current_model(obj.client(), service)
    if previous == model:
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": model, "changed": False})
            return
        render.success(f"'{service}' already routes to {model}")
        return
    if not _confirm(obj, yes, f"Switch '{service}' from {previous} to {model}?"):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": previous, "changed": False})
            return
        raise click.Abort()
    with render.status(f"Switching '{service}' to {model}…"):
        obj.client().set_model_service_model(service, model)
    if obj.output == "json":
        render.emit_json(
            {"model_service": service, "previous_model": previous, "model": model, "changed": True}
        )
        return
    render.success(
        f"Switched '{service}' to {model}",
        fields={"Previous model": previous},
    )
