"""Read-only project inventory and conservative project cleanup."""

from __future__ import annotations

import configparser
import os
import pathlib
from typing import Any
from urllib.parse import quote

import click
from databricks.sdk.errors import NotFound

from databricks_agentbricks import lakebase_runtime_store, render
from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli.deploy import (
    _confirm_destroy,
    _prefixed_name,
    _resolve_memory_store,
)
from databricks_agentbricks.cli.tracing import _mlflow, _set_tracking_uri, experiment_url
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_resources import load_receipts, save_receipts


def _source_option(function):
    return click.option(
        "--source",
        default=".",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        help="Project directory containing agent.toml (default: current directory).",
    )(function)


def _configured_host(profile: str | None) -> str | None:
    """Read only the chosen profile's host, without initializing auth or refreshing tokens."""
    if not profile:
        return None
    config = configparser.ConfigParser()
    config.read(os.getenv("DATABRICKS_CONFIG_FILE", str(pathlib.Path.home() / ".databrickscfg")))
    return config.get(profile, "host", fallback=None)


def inventory(source: pathlib.Path, *, profile: str | None, host: str | None) -> dict[str, Any]:
    project = AgentProject.load(source)
    receipts = load_receipts(source)
    resources: list[dict[str, Any]] = []

    def add(kind: str, name: str | None, action: str | None = None) -> None:
        if not name:
            return
        receipt = next(
            (
                r
                for r in receipts
                if r["host"] == (host or "").rstrip("/") and r["kind"] == kind and r["name"] == name
            ),
            None,
        )
        resources.append(
            {
                "kind": kind,
                "name": name,
                "id": receipt["id"] if receipt else None,
                "ownership": "created" if receipt and receipt["created"] else "adopted_or_unknown",
                "verification": "not_checked",
                "url": None,
                "next_action": action,
                "cleanup": receipt.get("cleanup", "active") if receipt else None,
            }
        )

    add(
        "deployment",
        _prefixed_name(project.deployment_name) if project.deployment_name else None,
        "Run agentbricks deploy to create or update the deployment.",
    )
    add(
        "memory_store",
        project.memory_store,
        "Run agentbricks deploy to provision the declared store.",
    )
    add(
        "session_store",
        project.session_store,
        "Run agentbricks deploy to provision the declared store.",
    )
    add(
        "experiment",
        project.trace_experiment_name,
        "Run agentbricks deploy to resolve the tracing experiment.",
    )
    for tool in project.tools:
        add(
            f"tool:{tool.source.kind}",
            tool.source.service or tool.source.function or tool.source.space_id or tool.id,
            "Tool access is verified when invoked; binding alone does not establish readiness.",
        )
    # Include earlier deployment names and stores after the manifest changes, so orphaned resources
    # are visible. Receipts from other workspaces must not appear as this workspace's resources.
    for receipt in receipts:
        if receipt["host"] == (host or "").rstrip("/") and not any(
            r["kind"] == receipt["kind"] and r["name"] == receipt["name"] for r in resources
        ):
            add(receipt["kind"], receipt["name"])
            resources[-1]["previous_binding"] = True
    return {
        "schema_version": 1,
        "project": str(source.resolve()),
        "framework": project.framework.value,
        "server": project.server.value,
        "profile": profile,
        "workspace": host,
        "verification": "not_checked",
        "resources": resources,
        "next_actions": (
            ["Select --profile <name> to identify the workspace."] if not profile else []
        )
        + (
            ["Run agentbricks deploy <name> to name and deploy this project."]
            if not project.deployment_name
            else []
        )
        + [
            "Run agentbricks status --verify for read-only resource checks.",
            "Resource availability does not verify model/tool invocation or the deployed app's permissions.",
        ],
    }


def verify_inventory(document: dict[str, Any], client: Any, profile: str) -> None:
    for resource in document["resources"]:
        kind, name = resource["kind"], resource["name"]
        try:
            if kind == "deployment":
                app = client.workspace_client.apps.get(name)
                resource["id"] = app.service_principal_client_id
                resource["url"] = app.url or f"{client.host}/apps/{quote(name, safe='')}"
                state = getattr(getattr(app, "app_status", None), "state", None)
                resource["state"] = getattr(state, "value", state)
            elif kind == "memory_store":
                store = _resolve_memory_store(client, name)
                if store is None:
                    raise AgentCliError(
                        "Store was not found in accessible stores.", error_code="NOT_FOUND"
                    )
                resource["id"] = render.field(store, "name")
            elif kind == "session_store":
                resource["id"] = render.field(client.get_session_store(name), "name")
            elif kind == "experiment":
                mlflow = _mlflow()
                _set_tracking_uri(mlflow, profile)
                experiment = mlflow.get_experiment_by_name(name)
                if experiment is None or experiment.lifecycle_stage == "deleted":
                    raise AgentCliError("Experiment does not exist.", error_code="NOT_FOUND")
                resource["id"] = experiment.experiment_id
                resource["url"] = experiment_url(client.host, experiment.experiment_id)
            else:
                continue
            resource["verification"] = "accessible"
            resource["next_action"] = None
        except Exception as exc:  # noqa: BLE001 - report partial results without losing the inventory
            resource["verification"] = (
                "missing"
                if isinstance(exc, NotFound)
                or getattr(exc, "error_code", None) in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}
                else "error"
            )
            resource["error"] = str(exc)
            if resource["verification"] == "error":
                resource["next_action"] = (
                    "Resolve the reported authentication/permission/service error and retry status --verify."
                )
    document["verification"] = "checked"
    document["next_actions"] = [a for a in document["next_actions"] if "status --verify" not in a]


def _show(document: dict[str, Any], output: str) -> None:
    if output == "json":
        render.emit_json(document)
        return
    click.echo(f"Project: {document['project']}")
    click.echo(f"Profile: {document['profile'] or 'not selected'}")
    click.echo(f"Workspace: {document['workspace'] or 'unresolved'}")
    for resource in document["resources"]:
        click.echo(
            f"  {resource['kind']}: {resource['name']} [{resource['verification']}; {resource['ownership']}]"
        )
        for key in ("id", "url", "state", "error", "next_action"):
            if resource.get(key):
                click.echo(f"    {key}: {resource[key]}")
    for action in document["next_actions"]:
        click.echo(f"  Next: {action}")


@click.command()
@_source_option
@click.option(
    "--verify", is_flag=True, help="Check resource availability using read-only workspace APIs."
)
@click.pass_obj
def status(obj, source: pathlib.Path, verify: bool) -> None:
    """Show project bindings and resources; configuration alone is not verified readiness."""
    client = None
    if verify:
        if not obj.profile:
            raise AgentCliError("Select --profile <name> before verifying workspace resources.")
        client = obj.client()
    host = client.host if client is not None else _configured_host(obj.profile)
    document = inventory(source, profile=obj.profile, host=host)
    if client is not None:
        verify_inventory(document, client, obj.profile)
    _show(document, obj.output)


def cleanup_plan(document: dict[str, Any], receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan = []
    for resource in document["resources"]:
        receipt = next(
            (
                r
                for r in receipts
                if r["host"] == document["workspace"].rstrip("/")
                and r["kind"] == resource["kind"]
                and r["name"] == resource["name"]
            ),
            None,
        )
        removable = bool(receipt and receipt["created"] and resource["kind"] == "deployment")
        plan.append(
            {
                "kind": resource["kind"],
                "name": resource["name"],
                "action": "delete" if removable else "retain",
                "reason": "Created by this project; app identity must still match before deletion."
                if removable
                else "Shared-capable resource or no creation receipt; retained to protect other consumers.",
                "runtime_store": "delete owner-validated managed store"
                if removable and receipt and receipt.get("managed_runtime")
                else "retain",
                "result": receipt.get("cleanup", "active") if receipt else "retained",
            }
        )
    return plan


def apply_cleanup(
    source: pathlib.Path, plan: list[dict[str, Any]], receipts: list[dict[str, Any]], client: Any
) -> bool:
    failed = False
    for item in plan:
        if item["action"] != "delete" or item["result"] == "deleted":
            continue
        receipt = next(
            r
            for r in receipts
            if r["host"] == client.host.rstrip("/")
            and r["kind"] == item["kind"]
            and r["name"] == item["name"]
        )
        name = receipt["name"]
        try:
            try:
                app = client.workspace_client.apps.get(name)
            except NotFound:
                # Without the app, there is no current owner to validate. Never delete a residual
                # Runtime Store just because its name matches the receipt.
                item["result"] = "app_missing"
                item["reason"] = (
                    "App already absent; any remaining Runtime Store is retained for manual inspection."
                )
            else:
                if app.service_principal_client_id != receipt["id"]:
                    raise AgentCliError(
                        "App identity changed since creation; retaining the app and Runtime Store."
                    )
                if receipt.get("managed_runtime"):
                    lakebase_runtime_store.delete(client, name, receipt["id"])
                client.workspace_client.apps.delete(name)
                item["result"] = "deleted"
            receipt["cleanup"] = item["result"]
            receipt.pop("cleanup_error", None)
        except Exception as exc:  # noqa: BLE001 - keep independent resources and retries usable
            failed = True
            item["result"] = "failed"
            item["error"] = str(exc)
            receipt["cleanup"] = "failed"
            receipt["cleanup_error"] = str(exc)
        save_receipts(source, receipts)
    return failed


@click.command()
@_source_option
@click.option("--apply", is_flag=True, help="Apply the previewed cleanup; default is preview only.")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt (requires --apply).")
@click.pass_obj
def cleanup(obj, source: pathlib.Path, apply: bool, yes: bool) -> None:
    """Preview cleanup of project-created deployments; retain shared stores, traces and tools."""
    if yes and not apply:
        raise AgentCliError("--yes requires --apply; cleanup is preview-only by default.")
    if not obj.profile:
        raise AgentCliError("Select --profile <name> to scope cleanup to one workspace.")
    client = obj.client() if apply else None
    host = client.host if client is not None else _configured_host(obj.profile)
    if not host:
        raise AgentCliError("Could not resolve the selected profile's workspace host.")
    document = inventory(source, profile=obj.profile, host=host)
    receipts = load_receipts(source)
    plan = cleanup_plan(document, receipts)
    if apply:
        # Always show the exact destructive scope before confirmation, including when JSON is the
        # requested final output. stderr keeps stdout machine-readable.
        targets = [p["name"] for p in plan if p["action"] == "delete" and p["result"] != "deleted"]
        click.echo(
            f"Workspace: {host}\nDelete: {', '.join(targets) or '(none)'}\nShared data, tools, source files and local state are retained.",
            err=True,
        )
        if targets:
            _confirm_destroy(
                "Delete these project-created deployments and their managed Runtime Stores",
                assume_yes=yes,
                err=True,
            )
        failed = apply_cleanup(source, plan, receipts, client)
    else:
        failed = False
    result = {"workspace": host, "profile": obj.profile, "applied": apply, "resources": plan}
    if obj.output == "json":
        render.emit_json(result)
    else:
        click.echo(f"Cleanup {'results' if apply else 'preview'} — {host}")
        for item in plan:
            click.echo(
                f"  {item['action']}: {item['kind']} {item['name']} — {item['result']}. {item['reason']}"
            )
            if item.get("error"):
                click.echo(f"    {item['error']}")
        if not apply:
            click.echo(
                "Run cleanup --apply to review and confirm deletion. No resources were changed."
            )
    if failed:
        raise click.exceptions.Exit(1)
