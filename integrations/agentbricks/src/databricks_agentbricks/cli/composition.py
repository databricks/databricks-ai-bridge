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
from databricks_agentbricks.services.deploy_service import DeployService
from databricks_agentbricks.services.provisioners import (
    AppProvisioner,
    MemoryStoreProvisioner,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)
from databricks_agentbricks.store_client import (
    MemoryStoreClient,
    RuntimeStoreClient,
    SessionStoreClient,
)


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
    api_client_factory = obj.client
    # One AppsClient for the whole command: its service-principal cache then collapses the memory
    # grant, session grant, and managed Runtime Store lookups into a single `apps get`.
    apps_client = AppsClient(obj.profile, runner=_databricks)
    reporter = ClickReporter()
    return DeployService(
        project_resolver=ProjectResolver(),
        apps_client=apps_client,
        api_client_factory=api_client_factory,
        app_provisioner=AppProvisioner(apps_client, api_client_factory, reporter),
        app_auth_client=AppAuthClient(obj.profile),
        memory_store_provisioner=MemoryStoreProvisioner(
            MemoryStoreClient(api_client_factory, apps_client), reporter
        ),
        session_store_provisioner=SessionStoreProvisioner(
            SessionStoreClient(api_client_factory, apps_client), reporter
        ),
        tracing_provisioner=TracingProvisioner(
            TracingClient(api_client_factory, obj.profile), reporter
        ),
        runtime_store_provisioner=RuntimeStoreProvisioner(
            RuntimeStoreClient(
                api_client_factory, obj.profile, apps_client, _USE_MANAGED_RUNTIME_STORE
            ),
            reporter,
        ),
        profile=obj.profile,
        reporter=reporter,
        prompter=ClickPrompter(),
    )
