"""Helpers for the live integration tests (imported by the tests; fixtures live in conftest.py)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic

from databricks_anthropic import (
    DatabricksSandboxDispatcher,
    DatabricksSandboxLauncher,
    LaunchResult,
)

ANTHROPIC_API_URL = "https://api.anthropic.com"
MAX_REPLY_WORDS = 25


def read_env_file(path: str | None) -> dict[str, str]:
    """Minimal dotenv reader: KEY=VALUE lines, optional quotes, # comments; a missing path is empty."""
    values: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            values[key.strip()] = value.split(" #", 1)[0].strip().strip("'\"")
    return values


def model() -> str:
    return os.environ.get("SANDBOX_E2E_MODEL", "claude-haiku-5-5")


def budget_cents() -> int:
    return int(os.environ.get("SANDBOX_E2E_BUDGET_CENTS", "100"))


RUN_PREFIX = f"cmat{uuid.uuid4().hex[:6]}-"


def find_or_create_agent(
    api: anthropic.Anthropic, *, variant: str, system: str, tools: list[Any]
) -> str:
    """Reuse the test agent for (variant, model) found by metadata, else create it (agents cost nothing)."""
    tag = {"purpose": "databricks-sandbox-e2e", "variant": variant, "model": model()}
    for agent in api.beta.agents.list(limit=100):
        if (agent.metadata or {}) == tag and not agent.archived_at:
            return agent.id
    agent = api.beta.agents.create(
        name=f"databricks-sandbox-e2e-{variant}",
        model=model(),
        system=system,
        tools=tools,
        metadata=tag,
    )
    print(f"\ncreated test agent {agent.id} ({variant}, {model()})")
    return agent.id


def make_launcher(workspace_client, settings, **kwargs) -> DatabricksSandboxLauncher:
    # A run-unique prefix: these tests can only ever create, find or clean up their own sandboxes.
    options: dict[str, Any] = {"max_idle_s": 60, "sandbox_id_prefix": RUN_PREFIX, **kwargs}
    return DatabricksSandboxLauncher(
        workspace_client=workspace_client,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        environment_key=settings["ANTHROPIC_ENVIRONMENT_KEY"],
        anthropic_base_url=settings.get("ANTHROPIC_BASE_URL") or ANTHROPIC_API_URL,
        **options,
    )


@dataclass
class Turn:
    reply: str = ""
    tool_calls: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    launches: list[LaunchResult] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SessionRun:
    """Single-turn result (kept for the single-turn tests)."""

    session_id: str
    sandbox_name: str
    reply: str = ""
    tool_calls: list[str] = field(default_factory=list)
    stop_reason: str | None = None
    launched: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    usage: dict | None = None


class SessionDriver:
    """One real session; each ``turn()`` sends a message and runs the dispatcher for *this* session only."""

    def __init__(
        self,
        settings,
        launcher: DatabricksSandboxLauncher,
        agent_id: str,
        *,
        resources: list[dict] | None = None,
        title: str = "databricks-sandbox e2e test",
    ):
        self.settings, self.launcher, self.agent_id = settings, launcher, agent_id
        self.resources, self.title = resources, title
        self._api: anthropic.AsyncAnthropic | None = None
        self.session_id = ""

    @property
    def api(self) -> anthropic.AsyncAnthropic:
        assert self._api is not None, "use SessionDriver as an async context manager"
        return self._api

    @property
    def sandbox_name(self) -> str:
        return self.launcher.sandbox_name_for(self.session_id)

    async def __aenter__(self) -> SessionDriver:
        self._api = anthropic.AsyncAnthropic(
            api_key=self.settings["ANTHROPIC_API_KEY"], base_url=self.launcher.anthropic_base_url
        )
        kwargs: dict[str, Any] = {"resources": self.resources} if self.resources else {}
        session = await self.api.beta.sessions.create(
            agent=self.agent_id,
            environment_id=self.settings["ANTHROPIC_ENVIRONMENT_ID"],
            title=self.title,
            budget={
                "type": "limit",
                "max_list_cost": {"amount": str(budget_cents()), "currency": "USD"},
            },
            **kwargs,
        )
        self.session_id = session.id
        return self

    async def __aexit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            await self.api.beta.sessions.delete(self.session_id)
        await self.api.close()

    async def turn(self, prompt: str, *, timeout_s: float = 420) -> Turn:
        turn, ready = Turn(), asyncio.Event()

        async def read_events() -> None:
            events = await self.api.beta.sessions.events.stream(self.session_id)
            async with events:
                ready.set()
                async for event in events:
                    if event.type in ("agent.tool_use", "agent.custom_tool_use"):
                        turn.tool_calls.append(getattr(event, "name", ""))
                    elif event.type == "agent.message":
                        content = getattr(event, "content", None) or []
                        turn.reply = "".join(getattr(b, "text", "") for b in content)
                    elif event.type == "session.status_idle":
                        turn.stop_reason = getattr(
                            getattr(event, "stop_reason", None), "type", None
                        )
                        if turn.stop_reason == "budget_reached" or (
                            turn.stop_reason != "requires_action" and turn.reply
                        ):
                            return

        # asyncio.wait_for, not asyncio.timeout: the package supports Python 3.10.
        streamer = asyncio.create_task(asyncio.wait_for(read_events(), timeout_s))
        await asyncio.wait_for(ready.wait(), 30)
        await self.api.beta.sessions.events.send(
            self.session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": prompt}]}],
        )
        await asyncio.sleep(2)  # let the session enqueue its work item
        # Only launch *this* session; anything else in the queue is left for the real dispatcher.
        dispatcher = DatabricksSandboxDispatcher(
            self.launcher, drain=True, accept=lambda w: w.data.id == self.session_id
        )
        turn.launches = await dispatcher.run()
        turn.failures = dispatcher.failures
        await streamer
        return turn

    async def usage(self) -> dict | None:
        return (await self.api.beta.sessions.retrieve(self.session_id)).model_dump().get("usage")


async def run_session_async(
    settings,
    launcher,
    agent_id: str,
    prompt: str,
    *,
    timeout_s: float,
    while_open: Callable[[str], Any] | None = None,
) -> SessionRun:
    """``while_open(sandbox_name)`` runs before the session is deleted (which ends the worker and its sandbox)."""
    async with SessionDriver(settings, launcher, agent_id) as driver:
        turn = await driver.turn(prompt, timeout_s=timeout_s)
        if while_open is not None:
            await asyncio.to_thread(while_open, driver.sandbox_name)
        return SessionRun(
            session_id=driver.session_id,
            sandbox_name=driver.sandbox_name,
            reply=turn.reply,
            tool_calls=turn.tool_calls,
            stop_reason=turn.stop_reason,
            launched=[r.sandbox_name for r in turn.launches],
            failures=turn.failures,
            usage=await driver.usage(),
        )
