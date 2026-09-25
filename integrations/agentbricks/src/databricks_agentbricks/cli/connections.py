"""Create or bind governed external connections in ``agent.toml``."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import click
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import ConnectionInfo, ConnectionType

from databricks_agentbricks import render
from databricks_agentbricks.agent_project import AgentProject, ConnectionSpec
from databricks_agentbricks.connection_registration import register_connection_via_dcr
from databricks_agentbricks.errors import AgentCliError


def _workspace_client(profile: str | None) -> Any:
    from databricks_agentkit._api_client import _workspace_client as build_client

    return build_client(profile)


def _record(spec: ConnectionSpec) -> dict[str, str]:
    return {
        "name": spec.name,
        "uc_connection": spec.uc_connection,
        "transport": spec.transport,
        "principal": spec.principal,
    }


def _validate_connection(info: ConnectionInfo, spec: ConnectionSpec) -> None:
    if info.connection_type != ConnectionType.HTTP:
        raise AgentCliError(f"{spec.uc_connection!r} is not a UC HTTP Connection.")


def _existing_connection(workspace_client: Any, spec: ConnectionSpec) -> ConnectionInfo:
    try:
        info = workspace_client.connections.get(spec.uc_connection)
    except NotFound as exc:
        raise AgentCliError(f"UC Connection {spec.uc_connection!r} does not exist.") from exc
    except Exception as exc:
        raise AgentCliError(f"Could not read UC Connection {spec.uc_connection!r}.") from exc
    _validate_connection(info, spec)
    return info


def _emit(obj: Any, action: str, project: AgentProject, spec: ConnectionSpec) -> None:
    if obj.output == "json":
        render.emit_json(
            {
                "action": action,
                "manifest": str(project.path),
                "connection": _record(spec),
            }
        )
        return
    render.success(
        f"{action.capitalize()} connection {spec.name!r}",
        fields={
            "UC Connection": spec.uc_connection,
            "Transport": spec.transport,
            "Principal": spec.principal,
            "Manifest": str(project.path),
        },
    )


@click.group("auth")
def auth() -> None:
    """Configure authentication and governed external connections."""


@auth.group()
def connections() -> None:
    """Create or bind UC Connections used by agent code."""


def _source_option(command):
    return click.option(
        "--source",
        default=".",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        help="Project directory containing agent.toml. Defaults to the current directory.",
    )(command)


@connections.command("bind")
@click.argument("name")
@click.option("--uc-connection", required=True, help="Existing three-part UC Connection name.")
@click.option("--transport", required=True, type=click.Choice(["mcp", "http"]))
@click.option("--principal", required=True, type=click.Choice(["user", "app"]))
@_source_option
@click.pass_obj
def bind_connection(
    obj: Any,
    name: str,
    uc_connection: str,
    transport: Literal["mcp", "http"],
    principal: Literal["user", "app"],
    source: pathlib.Path,
) -> None:
    """Bind NAME to an existing UC HTTP Connection."""

    project = AgentProject.load(source)
    spec = ConnectionSpec(name, uc_connection, transport, principal)
    changed = project.add_connection(spec)
    _existing_connection(_workspace_client(obj.profile), spec)
    if changed:
        project.write()
    _emit(obj, "bound", project, spec)


@connections.command("create")
@click.argument("name")
@click.option("--url", required=True, help="OAuth-protected HTTPS API or MCP endpoint URL.")
@click.option("--transport", required=True, type=click.Choice(["mcp", "http"]))
@click.option("--oauth", required=True, type=click.Choice(["dcr"]))
@click.option("--principal", required=True, type=click.Choice(["user", "app"]))
@click.option("--parent", required=True, help="Catalog and schema for the new connection.")
@_source_option
@click.pass_obj
def create_connection(
    obj: Any,
    name: str,
    url: str,
    transport: Literal["mcp", "http"],
    oauth: Literal["dcr"],
    principal: Literal["user", "app"],
    parent: str,
    source: pathlib.Path,
) -> None:
    """Create a DCR-backed UC HTTP Connection and bind it as NAME."""

    del oauth
    fqn = f"{parent}.{name}"
    project = AgentProject.load(source)
    spec = ConnectionSpec(name, fqn, transport, principal)
    project.add_connection(spec)
    workspace = _workspace_client(obj.profile)
    created = register_connection_via_dcr(
        workspace,
        fqn=fqn,
        url=url,
        transport=transport,
    )
    _validate_connection(created, spec)
    try:
        project.write()
    except AgentCliError as exc:
        recovery = (
            f"ab auth connections bind {name} --uc-connection {fqn} "
            f"--transport {transport} --principal {principal}"
        )
        raise AgentCliError(
            f"UC Connection {fqn!r} was created, but its binding could not be written.",
            hint=f"The remote connection was left intact. Recover with: {recovery}",
        ) from exc
    _emit(obj, "created", project, spec)
