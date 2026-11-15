"""Offline fakes for the CLI tests: no Databricks or Anthropic call is ever made."""

from __future__ import annotations

from types import SimpleNamespace


class FakeWorkspace:
    """Just enough of a WorkspaceClient for the CLI and the launcher's constructor."""

    def __init__(self, user: str = "dev@example.com", host: str = "https://ws.example.com") -> None:
        self.current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name=user))
        self.config = SimpleNamespace(
            host=host,
            auth_type="databricks-cli",
            user_agent="test-agent",
            with_user_agent_extra=lambda *args: None,
        )
        self.sandbox = SimpleNamespace(list_sandboxes=lambda **kwargs: iter([]))
        self.api_client = None


class FakeStats:
    """Stands in for ``client.beta.environments.work.stats``: yields the scripted counts, then the last one."""

    def __init__(self, *values: int | Exception | None) -> None:
        self._values = list(values)
        self.calls = 0

    def stats(self, environment_id: str) -> SimpleNamespace:
        self.calls += 1
        value = self._values.pop(0) if len(self._values) > 1 else self._values[0]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(workers_polling=value)

    def client(self) -> SimpleNamespace:
        return SimpleNamespace(beta=SimpleNamespace(environments=SimpleNamespace(work=self)))


class FakeDispatcher:
    """Stands in for DatabricksSandboxDispatcher in CLI tests; ``run`` returns at once."""

    instances: list[FakeDispatcher] = []

    def __init__(self, launcher=None, **kwargs) -> None:
        self.launcher = launcher
        self.kwargs = kwargs
        self.failures: list[tuple[str, str]] = []
        FakeDispatcher.instances.append(self)

    async def run(self) -> list:
        return []

    def list_session_sandboxes(self) -> list:
        return []

    def cleanup(self, **kwargs) -> list:
        return []
