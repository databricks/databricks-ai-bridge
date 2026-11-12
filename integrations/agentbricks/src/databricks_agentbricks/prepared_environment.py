"""Publish source before an App exists; retain legacy preparation support.

Workspace files hold the publication record and immutable source archive. Lakebase
holds sandbox sessions and shared runtime invocations/events. Only non-secret
connection coordinates are published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import ExportFormat, ImportFormat

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog
from databricks_agentkit.sandbox.control import ControlStore
from databricks_agentkit.sandbox.environment import (
    ARTIFACT_FORMAT,
    PREPARED_FORMAT,
    PUBLISHED_FORMAT,
    SHARED_FORMAT,
    on_demand,
)
from databricks_agentkit.sandbox.environments import EnvironmentStore
from databricks_agentkit.sandbox.lakebox import Lakebox
from databricks_agentkit.sandbox.orchestrator import Orchestrator
from databricks_agentkit.sandbox.store import Registry

FORMAT = ARTIFACT_FORMAT


def manifest_path(workspace, name):
    if name.startswith("/"):
        return name.removeprefix("/Workspace")
    user = workspace.current_user.me().user_name
    return f"/Users/{user}/agentbricks_environments/{name}/environment.json"


def read_manifest(workspace, name):
    path = manifest_path(workspace, name)
    with workspace.workspace.download(path, format=ExportFormat.AUTO) as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise AgentCliError("Environment manifest exceeds 1 MiB.")
    value = json.loads(data)
    if value.get("format") not in {FORMAT, PUBLISHED_FORMAT, SHARED_FORMAT, PREPARED_FORMAT}:
        raise AgentCliError("Not an Agent Bricks environment manifest.")
    if value.get("workspace", "").rstrip("/") != workspace.config.host.rstrip("/"):
        raise AgentCliError("Environment belongs to a different workspace.")
    value["manifest_path"] = path
    return value


def write_manifest(workspace, manifest, *, overwrite):
    workspace.workspace.upload(
        manifest["manifest_path"],
        json.dumps(manifest, indent=2, default=str).encode(),
        format=ImportFormat.RAW,
        overwrite=overwrite,
    )


def services(workspace, manifest):
    if "runtime_store" in manifest:
        raise AgentCliError(
            "This publication uses an App-owned Runtime Store.",
            hint="Invoke or debug through its App; local SQL access is not required or granted.",
        )
    connection = manifest["lakebase"]
    registry = Registry(
        workspace,
        manifest["namespace"],
        endpoint=connection["endpoint"],
        host=connection["host"],
        database=connection["database"],
        schema=manifest["namespace"].replace("-", "_"),
    )
    registry.initialize()
    catalog_path = manifest.get("catalog_path") or str(
        Path(manifest["manifest_path"]).parent / "catalog"
    )
    control = ControlStore(registry, WorkspaceCatalog(workspace, catalog_path))
    control.initialize()
    environments = EnvironmentStore(control)
    environments.initialize()
    lakebox = Lakebox(
        workspace,
        Path.home() / ".cache/agentbricks/sandbox",
        manifest["namespace"],
        transport="https",
    )
    return (
        registry,
        control,
        environments,
        Orchestrator(registry, lakebox, Path("."), manifest["namespace"]),
    )


def prepare(workspace, manifest, *, progress=None):
    """Retry preparation from the saved source, never from a mutable working folder."""
    if manifest.get("format") == ARTIFACT_FORMAT:
        return publish_artifact(workspace, manifest, progress=progress)
    if on_demand(manifest):
        raise AgentCliError(
            "This environment installs on its first invocation.",
            hint="Deploy it with agentbricks deploy APP --environment ENV, then invoke the App. "
            "environment prepare is only for legacy prepared environments.",
        )
    manifest.update(status="preparing", error=None)
    write_manifest(workspace, manifest, overwrite=True)
    try:
        _, control, _, orchestrator = services(workspace, manifest)
        agent = control.agent(manifest["agent_id"], with_bundle=True)
        bundle = Bundle(bytes(agent["bundle_data"]), agent["source_hash"], agent["file_count"])
        prepared = orchestrator.prepare(
            manifest["preparation_id"],
            bundle=bundle,
            entrypoint=manifest["entrypoint"],
            progress=progress,
        )
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4000])
        write_manifest(workspace, manifest, overwrite=True)
        raise AgentCliError(
            "Sandbox environment preparation failed.",
            hint=f"{manifest['error']}\nRetry: agentbricks environment prepare {manifest['manifest_path']}",
        ) from exc
    manifest.update(
        status="ready",
        preparation=prepared,
        execution_identity={
            "kind": "sandbox_creator",
            "principal": workspace.current_user.me().user_name,
        },
    )
    write_manifest(workspace, manifest, overwrite=True)
    return manifest


def _source_archive(workspace, manifest, bundle=None):
    """Reconcile a lost upload reply by checksum; never accept changed source."""
    from databricks_agentkit.sandbox.bundle import bundle_contents
    from databricks_agentkit.sandbox.sources import SourceLoader

    path = manifest.get("source_archive") or str(
        Path(manifest["manifest_path"]).parent / "source.tar.gz"
    )
    if bundle is None:
        try:
            with workspace.workspace.download(path, format=ExportFormat.AUTO) as stream:
                data = stream.read(21 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != manifest["source_hash"]:
                raise AgentCliError("Saved source archive digest mismatch")
            return Bundle(data, manifest["source_hash"], len(bundle_contents(data)))
        except NotFound:
            # Recover an interrupted first archive upload only if the Workspace
            # source reconstructs exactly the original immutable bundle.
            bundle = SourceLoader(workspace, Path(".")).load("workspace", manifest["source_path"])
    if bundle.digest != manifest["source_hash"]:
        raise AgentCliError("Source changed since creation; publish a new environment")
    try:
        workspace.workspace.upload(path, bundle.data, format=ImportFormat.RAW, overwrite=False)
    except Exception as original:
        # Upload can succeed even if its response was lost. Do not overwrite an
        # existing archive; accept only an exact digest readback.
        try:
            with workspace.workspace.download(path, format=ExportFormat.AUTO) as stream:
                saved = stream.read(21 * 1024 * 1024)
            if hashlib.sha256(saved).hexdigest() != bundle.digest:
                raise AgentCliError("Saved source archive digest mismatch")
        except NotFound:
            raise original from None
    return bundle


def publish_artifact(workspace, manifest, *, bundle=None, git_source=None, progress=None):
    """Retry a build using the frozen source; only the lock holder writes state."""
    if manifest.get("status") == "published":
        return manifest
    bundle = _source_archive(workspace, manifest, bundle)
    registry, control, environments, orchestrator = services(workspace, manifest)
    # A rejected concurrent attempt must never write its stale manifest. The
    # lock holder alone records build/failure state and cleanup intent.
    with registry.session("artifact-build:" + manifest["namespace"]):
        current = read_manifest(workspace, manifest["manifest_path"])
        manifest.clear()
        manifest.update(current)
        if manifest.get("status") == "published":
            return manifest
        return _publish_artifact(
            workspace, manifest, bundle, control, environments, orchestrator, progress
        )


def _publish_artifact(workspace, manifest, bundle, control, environments, orchestrator, progress):
    from databricks_agentbricks.services.environment_artifact import build_artifact

    try:
        try:
            row = environments.get(manifest["name"])
        except KeyError:
            row = None
        if row is not None:
            agent = control.agent(row["default_agent_id"])
            if agent["source_hash"] != manifest["source_hash"] or not agent.get(
                "execution_artifact"
            ):
                raise AgentCliError("Saved environment revision differs from this publication")
            manifest.update(
                agent_id=agent["id"],
                execution_artifact=agent["execution_artifact"],
                status="published",
                error=None,
            )
            write_manifest(workspace, manifest, overwrite=True)
            return manifest
        artifact = build_artifact(
            workspace,
            manifest,
            bundle,
            orchestrator.lakebox,
            lambda: write_manifest(workspace, manifest, overwrite=True),
            progress=progress,
        )
        row = environments.create(
            manifest["name"],
            "workspace",
            manifest["source_path"],
            manifest["entrypoint"],
            bundle=bundle,
            git_source=manifest.get("git_source"),
            freeze=True,
            execution_artifact=artifact,
        )
        manifest.update(agent_id=row["default_agent_id"], status="published", error=None)
        write_manifest(workspace, manifest, overwrite=True)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4000])
        write_manifest(workspace, manifest, overwrite=True)
        raise
    return manifest
