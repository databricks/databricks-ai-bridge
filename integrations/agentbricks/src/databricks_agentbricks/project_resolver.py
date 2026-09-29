"""Resolve an agent's authoring source (agent.toml): its project, bindings, and deployment name.

Render-free collaborator reading agent.toml so a service can resolve a project's bindings without
importing ``click`` or ``render``. agent.toml is the single source of truth for an agent's resources.
"""

from __future__ import annotations

import pathlib
from typing import Optional

from databricks_agentbricks.errors import AgentCliError


class ProjectResolver:
    """Read agent.toml at a source directory: load the project, its bindings, and its deployment name."""

    def load(self, source: pathlib.Path):
        """The AgentProject at `source`, or None when agent.toml is absent."""
        from databricks_agentbricks.agent_project import AgentProject

        if not (source / "agent.toml").is_file():
            return None
        return AgentProject.load(source)

    def resource_bindings(
        self, source: pathlib.Path
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """The (memory store, session store, tracing experiment) bound in agent.toml.

        agent.toml is the single source of truth for an agent's resources. Both `agentbricks dev` and `agentbricks deploy`
        resolve through here so the resource env/notices AND the deploy-time provisioning honor the
        same bindings. A missing agent.toml means nothing is bound; an invalid manifest fails with a clear
        error.
        """
        project = self.load(source)
        if project is None:
            return None, None, None
        # str(): agent.toml bindings come back as tomlkit strings, which don't serialize to app.yaml.
        memory = str(project.memory_store) if project.memory_store else None
        session = str(project.session_store) if project.session_store else None
        experiment = str(project.trace_experiment_name) if project.trace_experiment_name else None
        return memory, session, experiment

    def resolve_deployment_name(self, project, name: Optional[str]) -> str:
        """The deployment's base name: the NAME arg if given, else agent.toml's [agent].deployment_name.

        Errors when neither is available, pointing the user at the one-time `agentbricks deploy <name>`.
        """
        if name is not None and name.strip():
            return name.strip()
        if project is not None and project.deployment_name:
            return str(project.deployment_name)
        raise AgentCliError(
            "No deployment name given and none recorded in agent.toml.",
            hint="Run `agentbricks deploy <name>` once to name the agent; later `agentbricks deploy` can omit it.",
        )
