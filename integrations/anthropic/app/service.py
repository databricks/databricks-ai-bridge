"""Service mode (M1): the App's service principal serves every session of one self-hosted environment.

Configuration comes from the App's environment (``databricks.yml`` → ``config.env``). Secrets arrive through
App secret resources (``value_from``), so they are never in the bundle or the source:

  ANTHROPIC_ENVIRONMENT_ID      the service-mode environment ("S")                       (required)
  ANTHROPIC_ENVIRONMENT_KEY     its environment key (secret resource)                    (required)
  CMA_SANDBOX_ID_KEY            HMAC key for unpredictable sandbox names (secret)        (required)
  CMA_SANDBOX_PREFIX            sandbox id prefix (default "cmaapp-"); scopes cleanup
  CMA_MCP                       "genie-one" to serve Genie One from the sandbox, or empty
  CMA_LIFECYCLE                 "ephemeral" (default) or "resumable"
  CMA_EXPORT_VOLUME             optional UC Volume folder for outputs/
  CMA_MAX_CONCURRENT_LAUNCHES   default 8
  CMA_MAX_IDLE                  worker idle seconds before it returns (default 300)
  ANTHROPIC_BASE_URL_OVERRIDE   https only; default https://api.anthropic.com

The App's Databricks identity is its service principal (``WorkspaceClient()`` picks up the injected
``DATABRICKS_CLIENT_ID``/``SECRET``), so every sandbox, Genie One call and UC read acts as that principal.
The environment key stays in this process: sandboxes only ever get their own session's token.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from databricks_anthropic import DatabricksSandboxDispatcher, DatabricksSandboxLauncher
from databricks_anthropic.sandbox.launcher import DEFAULT_ANTHROPIC_BASE_URL

log = logging.getLogger("cma_dispatcher_app")

REQUIRED = ("ANTHROPIC_ENVIRONMENT_ID", "ANTHROPIC_ENVIRONMENT_KEY", "CMA_SANDBOX_ID_KEY")
# A crashed poll loop restarts after this backoff (doubling, capped).
RESTART_MIN_S, RESTART_MAX_S = 2.0, 60.0
# /api/health fails after this many consecutive poll-loop crashes, until a run stays up for RECOVERED_AFTER_S.
UNHEALTHY_AFTER_RESTARTS = 3
RECOVERED_AFTER_S = 2 * RESTART_MAX_S


class ConfigError(RuntimeError):
    """The App is missing required configuration (names only; never values)."""


def build_launcher(
    env: Mapping[str, str], workspace_client: WorkspaceClient | None = None
) -> DatabricksSandboxLauncher:
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        raise ConfigError(f"missing App configuration: {', '.join(missing)}")
    # Config first, then WorkspaceClient(config=...): the partner-guide pattern, so the client uses the SDK's
    # global product (set when databricks_anthropic is imported). WorkspaceClient(...) without a config
    # would default the product to unknown/0.0.0.
    w = workspace_client or WorkspaceClient(config=Config())
    mcp_servers = []
    mcp = (env.get("CMA_MCP") or "").strip()
    if mcp:
        if mcp != "genie-one":
            raise ConfigError(f"CMA_MCP must be 'genie-one' or empty, not {mcp!r}")
        from databricks_anthropic.sandbox.mcp import genie_one_server

        mcp_servers.append(genie_one_server(w))
    return DatabricksSandboxLauncher(
        workspace_client=w,
        environment_id=env["ANTHROPIC_ENVIRONMENT_ID"],
        environment_key=env["ANTHROPIC_ENVIRONMENT_KEY"],
        anthropic_base_url=env.get("ANTHROPIC_BASE_URL_OVERRIDE") or DEFAULT_ANTHROPIC_BASE_URL,
        sandbox_id_prefix=env.get("CMA_SANDBOX_PREFIX") or "cmaapp-",
        sandbox_id_key=env["CMA_SANDBOX_ID_KEY"],
        lifecycle=env.get("CMA_LIFECYCLE") or "ephemeral",  # type: ignore[arg-type]  # validated by the launcher
        export_volume_path=env.get("CMA_EXPORT_VOLUME") or None,
        max_idle_s=float(env.get("CMA_MAX_IDLE") or 300),
        mcp_servers=mcp_servers,
    )


@dataclass
class DispatcherService:
    """Runs the dispatcher for the App's lifetime, restarting the poll loop if it crashes."""

    launcher: DatabricksSandboxLauncher
    max_concurrent_launches: int = 8
    make_dispatcher: Callable[..., DatabricksSandboxDispatcher] = DatabricksSandboxDispatcher
    restarts: int = 0
    consecutive_crashes: int = 0
    last_crash: str | None = None
    started_at: float = field(default_factory=time.time)
    run_started_at: float | None = None  # time.monotonic() when the current poll loop started
    _task: asyncio.Task[None] | None = None
    dispatcher: DatabricksSandboxDispatcher | None = None
    _sandbox_counts: tuple[float, dict[str, int] | str] | None = None

    def start(self) -> None:
        self.dispatcher = self.make_dispatcher(
            self.launcher, max_concurrent_launches=self.max_concurrent_launches
        )
        self._task = asyncio.create_task(self._run_forever(), name="cma-dispatcher")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_forever(self) -> None:
        assert self.dispatcher is not None
        delay = RESTART_MIN_S
        while True:
            t0 = self.run_started_at = time.monotonic()
            try:
                await self.dispatcher.run()
                self.last_crash = "poll loop returned"
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep serving; report the type only
                self.last_crash = type(e).__name__
                log.error("dispatcher poll loop crashed: %s: %s", type(e).__name__, e)
            self.restarts += 1
            self.run_started_at = None
            # A loop that ran a while before failing is a fresh incident, not a crash loop.
            if time.monotonic() - t0 > RECOVERED_AFTER_S:
                self.consecutive_crashes, delay = 0, RESTART_MIN_S
            self.consecutive_crashes += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, RESTART_MAX_S)

    @property
    def healthy(self) -> bool:
        if self._task is None or self._task.done() or self.dispatcher is None:
            return False
        if not self.dispatcher.status.polling:
            return False
        if self.consecutive_crashes < UNHEALTHY_AFTER_RESTARTS:
            return True
        # After a crash loop: healthy again once the current run has stayed up long enough.
        running_for = time.monotonic() - self.run_started_at if self.run_started_at else 0.0
        if running_for > RECOVERED_AFTER_S:
            self.consecutive_crashes = 0
            return True
        return False

    def sandbox_counts(self, max_age_s: float = 30) -> dict[str, int] | str:
        """This deployment's sandboxes by state (its prefix and tags only), cached briefly.

        Only the App's identity can see them (``list_sandboxes`` is scoped to the caller), so this is how an
        operator confirms cleanup without access to the service principal.
        """
        now = time.monotonic()
        if self._sandbox_counts and now - self._sandbox_counts[0] < max_age_s:
            return self._sandbox_counts[1]
        counts: dict[str, int] | str
        try:
            counts = {}
            for sb in DatabricksSandboxDispatcher(self.launcher).list_session_sandboxes():
                state = sb.status.state.value if sb.status and sb.status.state else "UNKNOWN"
                state = state.removeprefix("SANDBOX_STATE_")
                counts[state] = counts.get(state, 0) + 1
        except Exception as e:  # noqa: BLE001 - status must not fail
            counts = f"unavailable ({type(e).__name__})"
        self._sandbox_counts = (now, counts)
        return counts

    def status(self) -> dict[str, Any]:
        """Counters only: no secrets, no session content, no error messages (types only)."""
        d = self.dispatcher
        return {
            "healthy": self.healthy,
            "mode": "service",
            "environment_id": self.launcher.environment_id,
            "sandbox_prefix": self.launcher.sandbox_id_prefix,
            "lifecycle": self.launcher.lifecycle,
            "mcp_servers": [s.name for s in self.launcher.mcp_servers],
            "uptime_s": round(time.time() - self.started_at),
            "restarts": self.restarts,
            "last_crash": self.last_crash,
            "dispatcher": d.status.as_dict() if d else None,
        }

    def full_status(self) -> dict[str, Any]:
        return {**self.status(), "sandboxes": self.sandbox_counts()}


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    return int(value) if value else default


def from_environment(env: Mapping[str, str] | None = None) -> DispatcherService:
    env = os.environ if env is None else env
    return DispatcherService(
        build_launcher(env), max_concurrent_launches=env_int(env, "CMA_MAX_CONCURRENT_LAUNCHES", 8)
    )
