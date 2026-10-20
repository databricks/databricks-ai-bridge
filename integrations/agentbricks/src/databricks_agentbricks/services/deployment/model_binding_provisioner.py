"""Reconcile bound model services and grant deployed Apps access to models and prompts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.provisioners import (
    GrantOutcome,
    ManifestPatch,
    ResourceContext,
)
from databricks_agentkit.runtime import prompt_registry
from databricks_agentkit.runtime.tool_manifest import model_service_env

_NOT_FOUND_CODES = {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}


@dataclass(frozen=True)
class ModelBindingState:
    """Bindings retained by the deploy service between reconciliation and access grants."""

    model_services: Mapping[str, str]
    prompts: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_services", MappingProxyType(dict(self.model_services)))
        object.__setattr__(self, "prompts", MappingProxyType(dict(self.prompts)))

    @property
    def manifest(self) -> ManifestPatch:
        return ManifestPatch(
            {model_service_env(role): name for role, name in self.model_services.items()}
        )


class ModelBindingProvisioner:
    """Create missing services without repointing existing ones; grant access after rollout."""

    def __init__(self, provider: ApiClientProvider, reporter: Reporter) -> None:
        self._provider = provider
        self._reporter = reporter

    def reconcile(self, ctx: ResourceContext) -> ModelBindingState:
        """Resolve each role to its service, creating only missing services with a default."""
        project = ctx.project.agent_project
        if project is None:
            return ModelBindingState({}, {})
        services: dict[str, str] = {}
        for role, binding in project.model_services.items():
            client = self._provider.get()
            name = binding.name
            with self._reporter.status(f"Reconciling model service '{name}' ({role})…"):
                try:
                    client.get_model_service(name)
                except AgentCliError as exc:
                    if exc.error_code not in _NOT_FOUND_CODES:
                        raise
                    if not binding.default:
                        raise AgentCliError(
                            f"Model service '{name}' doesn't exist and agent.toml declares no "
                            f"default model for role '{role}'.",
                            hint=f"Run `agentbricks experimental models bind {name} --role "
                            f"{role} --default system.ai.<model>` so deploy can create it.",
                        ) from exc
                    client.create_model_service(
                        name,
                        binding.default,
                        comment="Managed by agentbricks; repoint with `agentbricks experimental models`.",
                    )
                    self._reporter.note(f"Created model service {name!r} → {binding.default}")
            services[role] = name
        return ModelBindingState(services, project.prompts)

    def grant(
        self, state: ModelBindingState, principal: Optional[str]
    ) -> tuple[GrantOutcome, GrantOutcome]:
        """Grant each service and distinct prompt schema; retain errors for deploy output."""
        models = self._grant(tuple(state.model_services.values()), principal, prompts=False)
        schemas = tuple(sorted({prompt_registry.schema_of(p) for p in state.prompts.values()}))
        prompts = self._grant(schemas, principal, prompts=True)
        return models, prompts

    def _grant(
        self, names: tuple[str, ...], principal: Optional[str], *, prompts: bool
    ) -> GrantOutcome:
        if not names:
            return GrantOutcome.skipped()
        if principal is None:
            return GrantOutcome(True, "could not resolve the app's service principal.")
        failures: list[str] = []
        kind = "prompts" if prompts else "model services"
        with self._reporter.status(f"Granting the app access to its {kind}…"):
            client = self._provider.get()
            for name in names:
                try:
                    if prompts:
                        client.grant_prompt_access(name, principal)
                    else:
                        client.grant_model_service_execute(name, principal)
                except AgentCliError as exc:
                    failures.append(f"{name}: {exc.hint or exc}")
        return GrantOutcome(True, "; ".join(failures) or None)
