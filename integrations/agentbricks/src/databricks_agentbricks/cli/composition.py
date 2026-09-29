"""Wires the CLI context into a `DeployService`.

The service takes every collaborator through its constructor, which keeps it testable but means
somebody has to choose the real implementations. That choice lives here — one place where the CLI, and
only the CLI, assembles the object graph — so neither the command nor the service knows how the other
is built.
"""

from __future__ import annotations

from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.cli.presenter import ClickPrompter, ClickReporter
from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.manifest_manager import ManifestManager
from databricks_agentbricks.project_resolver import ProjectResolver
from databricks_agentbricks.runtime_store_provisioner import RuntimeStoreProvisioner
from databricks_agentbricks.services.deploy_service import DeployService
from databricks_agentbricks.store_provisioner import StoreProvisioner


def build_deploy_service(obj) -> DeployService:
    """A `DeployService` bound to the active CLI context (its profile, client, and terminal)."""
    # `TracingProvisioner` imports `cli.tracing`, and `cli/__init__` eagerly imports the command tree
    # (which reaches this module) - so a top-level import here would leave `tracing_provisioner`
    # un-importable on its own. Binding it at construction time keeps that module standalone.
    from databricks_agentbricks.tracing_provisioner import (  # noqa: PLC0415 - avoid import cycle
        TracingProvisioner,
    )

    runner = _databricks
    return DeployService(
        project=ProjectResolver(),
        apps=AppsClient(obj.profile, runner=runner),
        stores_factory=StoreProvisioner,
        manifest=ManifestManager(),
        tracing=TracingProvisioner(obj.profile),
        runtime_store=RuntimeStoreProvisioner(obj.profile),
        client_factory=obj.client,
        runner=runner,
        profile=obj.profile,
        reporter=ClickReporter(),
        prompter=ClickPrompter(),
    )
