"""Wires the CLI context into a `DeployService`.

The service takes every collaborator through its constructor, which keeps it testable but means
somebody has to choose the real implementations. That choice lives here — one place where the CLI, and
only the CLI, assembles the object graph — so neither the command nor the service knows how the other
is built.
"""

from __future__ import annotations

from databricks_agentbricks.app_auth_client import AppAuthClient
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.cli.presenter import ClickPrompter, ClickReporter
from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.deployment import _USE_MANAGED_RUNTIME_STORE
from databricks_agentbricks.project_resolver import ProjectResolver
from databricks_agentbricks.runtime_store_client import RuntimeStoreClient
from databricks_agentbricks.services.app_provisioner import AppProvisioner
from databricks_agentbricks.services.deploy_service import DeployService
from databricks_agentbricks.services.provisioners import (
    MemoryStoreProvisioner,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)
from databricks_agentbricks.store_client import MemoryStoreClient, SessionStoreClient


def build_deploy_service(obj) -> DeployService:
    """A `DeployService` bound to the active CLI context (its profile, client, and terminal)."""
    # `TracingClient` imports `cli.tracing`, and `cli/__init__` eagerly imports the command tree
    # (which reaches this module) - so a top-level import here would leave `tracing_client`
    # un-importable on its own. Binding it at construction time keeps that module standalone.
    from databricks_agentbricks.tracing_client import (  # noqa: PLC0415 - avoid import cycle
        TracingClient,
    )

    # `obj.client` memoizes the workspace client, so every collaborator handed this factory shares
    # the one instance - and, it being a factory, nothing opens the client before pre-flight.
    api = obj.client
    # One AppsClient for the whole command: its service-principal cache then collapses the memory
    # grant, session grant, and managed Runtime Store lookups into a single `apps get`.
    apps = AppsClient(obj.profile, runner=_databricks)
    reporter = ClickReporter()
    return DeployService(
        project=ProjectResolver(),
        apps_client=apps,
        api_client_factory=api,
        app=AppProvisioner(apps, api, reporter),
        app_auth=AppAuthClient(obj.profile),
        memory_store=MemoryStoreProvisioner(MemoryStoreClient(api, apps), reporter),
        session_store=SessionStoreProvisioner(SessionStoreClient(api, apps), reporter),
        tracing=TracingProvisioner(TracingClient(api, obj.profile), reporter),
        runtime_store=RuntimeStoreProvisioner(
            RuntimeStoreClient(api, obj.profile, apps, _USE_MANAGED_RUNTIME_STORE), reporter
        ),
        profile=obj.profile,
        reporter=reporter,
        prompter=ClickPrompter(),
    )
