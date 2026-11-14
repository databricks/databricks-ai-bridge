"""User mode (M2): sessions on the user environment ("U") run as the Databricks user who signs in.

Enabled with ``CMA_USER_MODE=1``. A user starts a session in the Claude Console on env U. One watcher polls env U (see
``signin.py``):

1. The session's first tool call finds no ticket, so the App answers it with a one-time ``/connect`` sign-in link
   instead of starting a sandbox.
2. The user opens the link. Databricks Apps signs them in, asks once for consent to the App's ``user_api_scopes`` and
   forwards their downscoped token in ``x-forwarded-access-token``. The page shows the session and agent; on
   confirmation (``POST /api/connect``) the App binds the session to the user's immutable Databricks id and leaves a
   ticket with the user's client.
3. The user replies "done". The next work item finds the ticket, and the sandbox is launched with the **user's**
   client, so the sandbox, its credential, Genie One and Unity Catalog reads act as that user.

State-changing requests need ``X-CMA-Request: 1`` and a same-host ``Origin`` (no cookies are used, so this is CSRF
defence in depth), and every body is ``extra="forbid"``. A session bound to another user answers 404. No response ever
contains the forwarded token.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anthropic
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from service import DispatcherService
from signin import (
    RoutingLauncher,
    Tickets,
    app_url_from_environment,
    code_hash,
    link_ticket_ttl,
    refusal,
    session_agent_id,
)
from user_store import Binding, LakebaseStore, Store, StoreUnavailable

from databricks_anthropic import DatabricksSandboxDispatcher, DatabricksSandboxLauncher
from databricks_anthropic.sandbox.launcher import DEFAULT_ANTHROPIC_BASE_URL
from databricks_anthropic.sandbox.mcp import genie_one_server

log = logging.getLogger("cma_dispatcher_app.user")

USER_MODE_ENV = "CMA_USER_MODE"
USER_SANDBOX_PREFIX = "cmausr-"
TOKEN_HEADER = "x-forwarded-access-token"
CSRF_HEADER = "x-cma-request"
STATIC = Path(__file__).resolve().parent / "static"  # the /connect page
PAGE_CSP = (
    "default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; "
    "frame-ancestors 'none'"
)


class ConfigError(RuntimeError):
    """User mode is misconfigured (names only; never values)."""


@dataclass(frozen=True)
class UserConfig:
    environment_id: str
    environment_key: str = field(repr=False)
    anthropic_api_key: str = field(repr=False)
    agents: tuple[str, ...]
    max_sessions_per_user: int

    @classmethod
    def from_environment(cls, env: Mapping[str, str], *, service_environment_id: str) -> UserConfig:
        required: dict[str, str] = {
            k: env.get(k) or ""
            for k in (
                "CMA_USER_ENVIRONMENT_ID",
                "ANTHROPIC_USER_ENVIRONMENT_KEY",
                "CMA_ANTHROPIC_API_KEY",
                "CMA_USER_AGENTS",
            )
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ConfigError(f"missing user-mode configuration: {', '.join(missing)}")
        agents = tuple(a.strip() for a in required["CMA_USER_AGENTS"].split(",") if a.strip())
        if not agents:
            raise ConfigError("CMA_USER_AGENTS lists no agents")
        # Env U must not be the service-mode environment: the service poller would serve user sessions as the
        # App's own principal.
        if required["CMA_USER_ENVIRONMENT_ID"] == service_environment_id:
            raise ConfigError("CMA_USER_ENVIRONMENT_ID must differ from ANTHROPIC_ENVIRONMENT_ID")
        return cls(
            environment_id=required["CMA_USER_ENVIRONMENT_ID"],
            environment_key=required["ANTHROPIC_USER_ENVIRONMENT_KEY"],
            anthropic_api_key=required["CMA_ANTHROPIC_API_KEY"],
            agents=agents,
            max_sessions_per_user=int(env.get("CMA_MAX_SESSIONS_PER_USER") or 20),
        )


@dataclass(frozen=True)
class Caller:
    principal_id: str  # the user's immutable Databricks id: the key for bindings
    user_name: str


class Identity:
    """Resolves a forwarded user token to a :class:`Caller`. Caches principals by token digest, never tokens."""

    def __init__(
        self,
        host: str,
        *,
        ttl_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        client_factory: Callable[[str, str], WorkspaceClient] | None = None,
    ) -> None:
        self._host = host
        self._ttl = ttl_s
        self._clock = clock
        self._make = client_factory or _user_client
        self._cache: dict[str, tuple[float, Caller]] = {}
        self._lock = threading.Lock()

    def resolve(self, token: str) -> tuple[Caller, WorkspaceClient]:
        """The caller and a client that acts as them. Raises if the token is not accepted."""
        client = self._make(self._host, token)
        key = hashlib.sha256(token.encode()).hexdigest()
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1], client
        me = client.current_user.me()
        caller = Caller(principal_id=me.id or "", user_name=me.user_name or "")
        if not caller.principal_id:
            raise PermissionError("no Databricks identity for this token")
        with self._lock:
            if len(self._cache) > 1000:
                self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            self._cache[key] = (now + self._ttl, caller)
        return caller, client


def _user_client(host: str, token: str) -> WorkspaceClient:
    # A PAT-style bearer with the forwarded token (verified live in the on-behalf-of spike, design doc §16 S3).
    return WorkspaceClient(config=Config(host=host, token=token, auth_type="pat"))


def user_launcher(
    service: DatabricksSandboxLauncher, config: UserConfig, client: WorkspaceClient
) -> DatabricksSandboxLauncher:
    """The service launcher's settings, retargeted at env U and the caller's identity.

    ``dataclasses.replace`` re-runs the launcher's validation. The sandbox id key and export settings carry over.
    """
    return dataclasses.replace(
        service,
        workspace_client=client,
        environment_id=config.environment_id,
        environment_key=config.environment_key,
        sandbox_id_prefix=USER_SANDBOX_PREFIX,
        mcp_servers=[genie_one_server(client)] if service.mcp_servers else [],
        lifecycle="ephemeral",
    )


@dataclass
class UserMode:
    """Everything the user-mode routes need. Built once per process (see :func:`user_mode_from_environment`)."""

    config: UserConfig
    identity: Identity
    store: Store
    anthropic: Any  # an anthropic.Anthropic client (or a fake with the same beta.sessions methods)
    service_launcher: DatabricksSandboxLauncher
    tickets: Tickets = field(default_factory=Tickets)
    app_url: str | None = None
    watcher: DispatcherService | None = None
    _agent_names: dict[str, str] = field(default_factory=dict)

    def agent_name(self, agent_id: str) -> str:
        """The agent's display name (looked up once, cached); the id itself if the lookup fails."""
        if agent_id not in self._agent_names:
            try:
                name = self.anthropic.beta.agents.retrieve(agent_id).name
            except Exception as e:  # noqa: BLE001 - fall back to the id; retry on the next call
                log.warning("agent name lookup failed for %s: %s", agent_id, type(e).__name__)
                return agent_id
            self._agent_names[agent_id] = name or agent_id
        return self._agent_names[agent_id]

    def launcher_for(self, client: WorkspaceClient) -> DatabricksSandboxLauncher:
        return user_launcher(self.service_launcher, self.config, client)

    def routing_launcher(self) -> RoutingLauncher:
        return RoutingLauncher(
            template=self.launcher_for(self.service_launcher.workspace_client),
            launcher_for=self.launcher_for,
            tickets=self.tickets,
            store=self.store,
            sessions_api=self.anthropic,
            agents=self.config.agents,
            app_url=self.app_url,
        )

    def start(self) -> None:
        """Start the env-U watcher (the only poller of env U)."""
        self.watcher = DispatcherService(
            self.routing_launcher(),  # type: ignore[arg-type] - quacks like a launcher for the dispatcher
            make_dispatcher=_watcher_dispatcher,
        )
        self.watcher.start()

    async def stop(self) -> None:
        if self.watcher:
            await self.watcher.stop()


def _watcher_dispatcher(
    launcher: Any, *, max_concurrent_launches: int
) -> DatabricksSandboxDispatcher:
    # No sweeps: user sandboxes belong to their users (the App's identity can't list them).
    return DatabricksSandboxDispatcher(
        launcher, max_concurrent_launches=max_concurrent_launches, sweep_interval_s=None
    )


def user_mode_from_environment(service: Any, env: Mapping[str, str] | None = None) -> UserMode:
    """Build user mode from the App's environment, next to the running service (its launcher is the template)."""
    env = os.environ if env is None else env
    launcher = service.launcher
    config = UserConfig.from_environment(env, service_environment_id=launcher.environment_id)
    return UserMode(
        config=config,
        identity=Identity(launcher.workspace_client.config.host),
        store=LakebaseStore.from_environment(dict(env), launcher.workspace_client),
        anthropic=anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            base_url=DEFAULT_ANTHROPIC_BASE_URL,
        ),
        service_launcher=launcher,
        app_url=app_url_from_environment(env, launcher.workspace_client),
    )


# ---------------------------------------------------------------- request models


class Connect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=16, max_length=100)


# ---------------------------------------------------------------- helpers


def _csrf_guard(request: Request) -> None:
    """State changes need a custom header (cross-site forms cannot set it) and a same-host Origin."""
    if request.headers.get(CSRF_HEADER) != "1":
        raise HTTPException(403, f"missing {CSRF_HEADER} header")
    origin = request.headers.get("origin") or ""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    parts = urlsplit(origin)
    if (
        not origin
        or not host
        or parts.netloc.lower() != host.lower()
        or parts.scheme not in ("https", "http")
    ):
        raise HTTPException(403, "cross-origin request refused")


def _token(request: Request) -> str:
    token = request.headers.get(TOKEN_HEADER)
    if not token:
        raise HTTPException(401, "sign in through the App URL to use user mode")
    return token


async def _caller(request: Request) -> tuple[Caller, WorkspaceClient]:
    user: UserMode = request.app.state.user
    token = _token(request)
    try:
        return await asyncio.to_thread(user.identity.resolve, token)
    except Exception as e:  # noqa: BLE001 - never echo the token or the SDK message
        log.info("rejected a user token: %s", type(e).__name__)
        raise HTTPException(
            401, "your Databricks sign-in could not be verified; reopen the App"
        ) from None


async def _db(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception as e:  # noqa: BLE001 - StoreUnavailable and anything else from the store
        log.warning(
            "store call failed: %s", type(e).__name__ if not isinstance(e, StoreUnavailable) else e
        )
        raise HTTPException(503, "the session store is unavailable; try again shortly") from None


async def _api(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001 - the Anthropic error text can echo request data
        status = getattr(e, "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        log.warning("anthropic call failed: %s%s", type(e).__name__, detail)
        raise HTTPException(
            502, f"the Claude API rejected the request: {type(e).__name__}{detail}"
        ) from None


def _owned(binding: Binding | None, caller: Caller) -> Binding:
    # Foreign and unknown sessions look the same: 404, so ids can't be probed.
    if binding is None or binding.principal_id != caller.principal_id:
        raise HTTPException(404, "no such session")
    return binding


# ---------------------------------------------------------------- routes


def _page(path: Path) -> HTMLResponse:
    return HTMLResponse(path.read_text(), headers={"Content-Security-Policy": PAGE_CSP})


def _script(path: Path) -> FileResponse:
    return FileResponse(
        path, media_type="application/javascript", headers={"Content-Security-Policy": PAGE_CSP}
    )


def add_routes(app: FastAPI) -> None:
    """Register the user-mode routes. Called only when CMA_USER_MODE=1 (the routes are absent otherwise)."""

    @app.get("/connect", response_class=HTMLResponse)
    async def connect_page():
        return _page(STATIC / "connect.html")

    @app.get("/connect.js")
    async def connect_js():
        return _script(STATIC / "connect.js")

    @app.get("/api/connect")
    async def connect_info(request: Request, code: str = Query(min_length=16, max_length=100)):
        """What a sign-in link is for, before the user confirms. Consumes nothing."""
        user: UserMode = request.app.state.user
        caller, _ = await _caller(request)
        session_id = await _db(user.store.peek_code, code_hash(code))
        if not session_id:
            raise HTTPException(404, "this sign-in link has expired or was already used")
        binding = await _db(user.store.binding, session_id)
        if binding is not None and binding.principal_id != caller.principal_id:
            raise HTTPException(404, "this sign-in link has expired or was already used")
        session = await _api(user.anthropic.beta.sessions.retrieve, session_id)
        agent_id = session_agent_id(session)
        return {
            "session_id": session_id,
            "agent": await asyncio.to_thread(user.agent_name, agent_id) if agent_id else None,
            "signed_in_as": caller.user_name,
            "already_yours": binding is not None,
            "problem": refusal(
                session, environment_id=user.config.environment_id, agents=user.config.agents
            ),
        }

    @app.post("/api/connect", dependencies=[Depends(_csrf_guard)])
    async def connect(body: Connect, request: Request):
        """Redeem a sign-in link: bind the session to the caller (unless already theirs) and leave a ticket."""
        user: UserMode = request.app.state.user
        caller, client = await _caller(request)
        hashed = code_hash(body.code)
        gone = HTTPException(404, "this sign-in link has expired or was already used")
        session_id = await _db(user.store.peek_code, hashed)
        if not session_id:
            raise gone
        binding = await _db(user.store.binding, session_id)
        if binding is not None and binding.principal_id != caller.principal_id:
            await _db(
                user.store.audit,
                caller.principal_id,
                session_id,
                "connect-refused",
                "bound to another user",
            )
            raise gone
        session = await _api(user.anthropic.beta.sessions.retrieve, session_id)
        problem = refusal(
            session, environment_id=user.config.environment_id, agents=user.config.agents
        )
        if problem:
            raise HTTPException(403, problem)
        if binding is None:
            count = await _db(user.store.count_for, caller.principal_id)
            if count >= user.config.max_sessions_per_user:
                raise HTTPException(429, "you have reached the session limit for user mode")
        if await _db(user.store.use_code, hashed, caller.principal_id) != session_id:
            raise gone  # redeemed by someone else in the meantime
        if binding is None:
            try:
                await asyncio.to_thread(
                    user.store.bind,
                    Binding(
                        session_id, caller.principal_id, caller.user_name, session_agent_id(session)
                    ),
                )
            except StoreUnavailable:
                # Bound concurrently: only fine if it is now ours.
                _owned(await _db(user.store.binding, session_id), caller)
        user.tickets.put(
            session_id, caller.principal_id, client, ttl_s=link_ticket_ttl(_token(request))
        )
        await _db(
            user.store.audit, caller.principal_id, session_id, "connect", session_agent_id(session)
        )
        log.info("session=%s connected through a sign-in link", session_id)
        return {
            "session_id": session_id,
            "note": "Connected. Go back to Claude and reply 'done'; the agent's tools now run as you.",
        }
