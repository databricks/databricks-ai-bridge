"""Publish source as part of deployment; compute belongs to the serving App."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tomli
from databricks.sdk.errors import NotFound
from databricks.sdk.service.workspace import ExportFormat, ImportFormat

from databricks_agentbricks.environment_source import (
    entrypoint_name,
    git_project_files,
    project_files,
    workspace_path,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.prepared_environment import read_manifest
from databricks_agentbricks.services.deployment.names import (
    DeploymentName,
    _prefixed_name,
    base_name_problem,
)
from databricks_agentkit.sandbox.bundle import bundle_files
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog
from databricks_agentkit.sandbox.environment import PUBLISHED_FORMAT, saved_bundle
from databricks_agentkit.sandbox.environments import EnvironmentExists, PublicationStore


@dataclass(frozen=True)
class SandboxDeployRequest:
    name: str | None
    source: Path
    sandbox_idle_timeout: int = 900
    git_revision: str | None = None
    entrypoint: str | None = None
    workspace_path: str | None = None
    instances: int | None = None
    index_url: str = "https://pypi.org/simple/"


class SandboxDeployService:
    def __init__(self, workspace, deployer: Callable):
        self.workspace = workspace
        self.deployer = deployer

    def deploy(self, request: SandboxDeployRequest):
        """Freeze one source revision, then deploy only the generic orchestrator."""
        if not 60 <= request.sandbox_idle_timeout <= 86400:
            raise AgentCliError("Sandbox idle timeout must be 60–86400 seconds.")
        git_source = None
        if request.git_revision is not None:
            if not request.git_revision.strip():
                raise AgentCliError("--git-revision must name a local commit, tag, or branch.")
            files, git_source = git_project_files(request.source, request.git_revision)
        else:
            files = project_files(request.source)
        project = tomli.loads(files.get("agent.toml", b"").decode())
        name = request.name or project.get("agent", {}).get("deployment_name")
        if not name:
            raise AgentCliError("Provide a deployment name or set [agent].deployment_name.")
        if problem := base_name_problem(name):
            raise AgentCliError(problem)
        name = DeploymentName(_prefixed_name(name))
        if any(tool.get("auth") == "user" for tool in project.get("tools", [])):
            raise AgentCliError(
                "Sandbox deployments do not forward request-user credentials.",
                hint="Configure managed tools with --auth app.",
            )
        entrypoint = request.entrypoint or (
            "runtime.sandbox_entry:invoke" if "runtime/sandbox_entry.py" in files else None
        )
        if not entrypoint:
            raise AgentCliError("Provide --entrypoint module:function or use a sandbox template.")
        entrypoint = entrypoint_name(entrypoint)
        workspace = self.workspace
        root = (
            workspace_path(request.workspace_path)
            if request.workspace_path
            else (f"/Users/{workspace.current_user.me().user_name}/agentbricks_deployments/{name}")
        )
        bundle = bundle_files(files)
        # Preserve the published identity format so redeploys keep revisions and sessions.
        identity = json.dumps(
            ["managed-runtime-store-v1", name, bundle.digest, entrypoint, git_source, None],
            sort_keys=True,
        )
        revision = hashlib.sha256(identity.encode()).hexdigest()
        publication = f"{name}-{revision[:16]}"
        manifest_path = f"{root}/revisions/{revision}/environment.json"
        namespace = (
            "ab-deploy-"
            + hashlib.sha256(
                f"{workspace.config.host.rstrip('/')}\n{root}\nmanaged-v1".encode()
            ).hexdigest()[:20]
        )
        manifest = {
            "format": PUBLISHED_FORMAT,
            "storage_version": 3,
            "catalog_path": f"{root}/catalog",
            "name": publication,
            "deployment": name,
            "workspace": workspace.config.host,
            "namespace": namespace,
            "entrypoint": entrypoint,
            "manifest_path": manifest_path,
            "source_path": f"{root}/revisions/{revision}/source",
            "source_hash": bundle.digest,
            "runtime_key": f"environment:{publication}",
            "sandbox_scope": "session",
            "git_source": git_source,
            "status": "published",
            "runtime_store": {"name": f"runtime-stores/{name}", "owner_app": name},
        }
        control = WorkspaceCatalog(workspace, manifest["catalog_path"])
        environments = PublicationStore(control, namespace)
        # Immutable Workspace writes reconcile concurrent/lost replies. Runtime
        # SQL is App-owned and is never opened by the source publisher.
        try:
            existing = read_manifest(workspace, manifest_path)
        except NotFound:
            existing = None
        if existing is not None:
            for key in (
                "format",
                "storage_version",
                "catalog_path",
                "name",
                "deployment",
                "namespace",
                "entrypoint",
                "source_path",
                "source_hash",
                "runtime_key",
                "sandbox_scope",
                "git_source",
                "status",
                "runtime_store",
            ):
                if existing.get(key) != manifest.get(key):
                    raise AgentCliError("Existing deployment revision metadata does not match.")
            if saved_bundle(control, existing).data != bundle.data:
                raise AgentCliError("Existing deployment revision has different source.")
            manifest = existing
        else:
            # The frozen bundle is authoritative; per-file readback makes partial
            # Workspace uploads retryable without overwriting another publication.
            for relative, contents in sorted(files.items()):
                _publish_file(workspace, f"{manifest['source_path']}/{relative}", contents)
            try:
                row = environments.get(publication)
            except KeyError:
                try:
                    row = environments.create(
                        publication,
                        "workspace",
                        manifest["source_path"],
                        entrypoint,
                        bundle=bundle,
                        git_source=git_source,
                        freeze=True,
                    )
                except EnvironmentExists:
                    # Another publisher won the Workspace write. Verify the
                    # saved source below rather than overwriting its record.
                    row = environments.get(publication)
            manifest["agent_id"] = row["default_agent_id"]
            agent = control.agent(manifest["agent_id"], with_bundle=True)
            if (
                agent["source_hash"],
                agent["entrypoint"],
                agent["source_path"],
                bytes(agent["bundle_data"]),
            ) != (
                bundle.digest,
                entrypoint,
                manifest["source_path"],
                bundle.data,
            ):
                raise AgentCliError("Saved deployment revision does not match source publication.")
            _publish_file(
                workspace,
                manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True).encode(),
            )
        # App sources are separate from the immutable revision directories.
        # Reusing this root preserves older revisions and session mappings.
        return self.deployer(
            name,
            manifest_path,
            workspace_path=f"{root}/app",
            instances=request.instances,
            sandbox_idle_timeout=request.sandbox_idle_timeout,
            index_url=request.index_url,
        )


def _publish_file(workspace, path: str, contents: bytes):
    def read():
        with workspace.workspace.download(path, format=ExportFormat.AUTO) as stream:
            return stream.read(len(contents) + 1)

    try:
        existing = read()
    except NotFound:
        workspace.workspace.mkdirs(str(Path(path).parent))
        try:
            workspace.workspace.upload(path, contents, format=ImportFormat.RAW, overwrite=False)
            return
        except Exception as original:
            # An accepted upload can lose its reply. Verify before retrying it.
            try:
                existing = read()
            except NotFound:
                raise original from None
    if existing != contents:
        raise AgentCliError(f"Published source differs at {path}; existing files were preserved.")
