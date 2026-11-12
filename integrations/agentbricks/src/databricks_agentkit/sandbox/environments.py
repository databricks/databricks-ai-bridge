"""Compatibility names for frozen Workspace publications; no environment SQL tables."""

import re
from datetime import datetime, timezone


class EnvironmentExists(ValueError):
    pass


def validate_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Environment names use 1–64 letters, digits, underscores, or hyphens")
    return name


class PublicationStore:
    """Workspace-only publication; the deploying user never needs runtime SQL access."""

    def __init__(self, catalog, namespace, sources=None):
        self.catalog, self.namespace, self.sources = catalog, namespace, sources

    def initialize(self):
        # Retained for callers of the experimental environment API.
        pass

    def create(
        self,
        name,
        kind,
        path,
        entrypoint,
        *,
        bundle=None,
        git_source=None,
        freeze=False,
        execution_artifact=None,
    ):
        validate_name(name)
        try:
            self.get(name)
        except KeyError:
            pass
        else:
            raise EnvironmentExists("Environment already exists; choose a new name")
        if bundle is None:
            if self.sources is None:
                raise ValueError("Publication requires source files")
            bundle = self.sources.load(kind, path)
        agent = self.catalog.create_agent(
            f"environment:{name}",
            kind,
            path,
            entrypoint,
            bundle,
            git_source=git_source,
            execution_artifact=execution_artifact,
        )
        value = dict(
            namespace=self.namespace,
            name=name,
            source_kind=kind,
            source_path=path,
            entrypoint=entrypoint,
            default_agent_id=agent["id"],
            git_source=git_source,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            return self.catalog.put("environments", name, value)
        except ValueError as exc:
            existing = self.get(name)
            if {k: v for k, v in existing.items() if k != "created_at"} == {
                k: v for k, v in value.items() if k != "created_at"
            }:
                return existing
            raise EnvironmentExists("Environment already exists; choose a new name") from exc

    def get(self, name):
        return self.catalog.get("environments", validate_name(name))

    def list(self):
        return self.catalog.list("environments")


class EnvironmentStore(PublicationStore):
    def __init__(self, control, sources=None):
        super().__init__(control.catalog, control.namespace, sources)
        self.control, self.registry = control, control.registry

    def session_agent(self, name, session_id):
        agent_id = self.get(name)["default_agent_id"]
        if not agent_id:
            raise ValueError("Publication has no frozen revision; publish it again")
        return agent_id

    def invoke(self, name, session_id, invocation_id, value, *, retry=False):
        agent_id = self.session_agent(name, session_id)
        return self.control.create_run(agent_id, session_id, invocation_id, value, retry=retry)
