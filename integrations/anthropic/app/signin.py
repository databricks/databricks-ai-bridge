"""User mode's single poller on the user environment ("U"), and the sign-in link.

Env U is polled by one watcher (a normal :class:`DatabricksSandboxDispatcher`) whose launcher routes each session's
work item:

- **A ticket is waiting** for that session: the user opened its sign-in link and confirmed, which left their client
  for it. If the session is bound to that same user, the sandbox is launched with the user's client, so it runs as
  them.
- **No ticket** (a session started in the Claude Console, or a bound session whose ticket has lapsed): the App
  answers the session's pending tool calls itself, with a one-time sign-in link to ``/connect``, and releases the
  item. Nothing runs in a sandbox. Once the user signs in through the link, the next tool call runs as them.

Tickets live in this process's memory only, for minutes (never in the store, never logged). A ticket holds a
client built from the user's forwarded token; it expires before the token does.

Sign-in codes are random (``secrets.token_urlsafe``), stored only as SHA-256 hashes, valid for ``CODE_TTL_S`` and
single-use (see ``user_store``). Whoever opens a link supplies the identity, so ``/connect`` shows which session
and agent the link is for and asks for confirmation; a session bound to another user can't be taken over.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
from databricks.sdk import WorkspaceClient
from user_store import Binding, Store

from databricks_anthropic import DatabricksSandboxLauncher
from databricks_anthropic.sandbox.launcher import sessions_token_from_secret

log = logging.getLogger("cma_dispatcher_app.signin")

CODE_TTL_S = 15 * 60
# A ticket from /connect lives until the user replies in the Console: until the token nears expiry, at most this.
LINK_TICKET_MAX_S = 30 * 60
TOKEN_MARGIN_S = 120.0
# How long the App keeps answering one item's tool calls with the link before releasing it.
LINK_ITEM_MAX_S = 300.0
LINK_POLL_S = 2.0
TOOL_USE_TYPES = ("agent.tool_use", "agent.custom_tool_use")
WATCH_EVENT_TYPES = (
    *TOOL_USE_TYPES,
    "user.tool_result",
    "user.custom_tool_result",
    "session.status_idle",
    "session.status_running",
    "session.status_terminated",
)


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def token_expiry(token: str, now: float | None = None) -> float | None:
    """Seconds until a JWT expires (its ``exp`` claim), or None if it isn't a JWT. The token is not kept."""
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(claims["exp"]) - (time.time() if now is None else now)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- tickets


@dataclass
class Ticket:
    principal_id: str
    client: WorkspaceClient = field(repr=False)
    expires_at: float  # time.monotonic()


class Tickets:
    """Session id -> the signed-in user's client, waiting for that session's next work item. Memory only."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._tickets: dict[str, Ticket] = {}

    def put(
        self, session_id: str, principal_id: str, client: WorkspaceClient, *, ttl_s: float
    ) -> Ticket:
        ticket = Ticket(principal_id, client, self._clock() + ttl_s)
        with self._lock:
            now = self._clock()
            self._tickets = {k: t for k, t in self._tickets.items() if t.expires_at > now}
            self._tickets[session_id] = ticket
        return ticket

    def get(self, session_id: str) -> Ticket | None:
        with self._lock:
            ticket = self._tickets.get(session_id)
            if ticket and ticket.expires_at <= self._clock():
                del self._tickets[session_id]
                return None
            return ticket

    def __len__(self) -> int:
        with self._lock:
            return len(self._tickets)


def link_ticket_ttl(token: str) -> float:
    """How long a path-B ticket may live: until shortly before the user's token expires, capped."""
    left = token_expiry(token)
    if left is None:
        return 15 * 60.0
    return max(0.0, min(LINK_TICKET_MAX_S, left - TOKEN_MARGIN_S))


# ---------------------------------------------------------------- session checks


def refusal(session: Any, *, environment_id: str, agents: tuple[str, ...]) -> str | None:
    """Why this session can't run in user mode, or None. Messages are shown to the user (no secrets)."""
    if getattr(session, "environment_id", None) != environment_id:
        return "this session is not on the user-mode environment"
    agent = getattr(session, "agent", None)
    agent_id = getattr(agent, "id", None) if agent is not None else None
    if agent_id not in agents:
        return "this agent is not enabled for Databricks user mode"
    if any(
        getattr(r, "type", None) == "memory_store"
        for r in getattr(session, "resources", None) or []
    ):
        return "sessions with memory stores are not supported in Databricks user mode"
    if getattr(session, "vault_ids", None):
        return "sessions with vaults are not supported in Databricks user mode"
    return None


def session_agent_id(session: Any) -> str:
    agent = getattr(session, "agent", None)
    return str(getattr(agent, "id", "") or "")


# ---------------------------------------------------------------- the routing launcher


@dataclass(frozen=True)
class Deferred:
    """What the watcher records for an item the App answered itself (no sandbox)."""

    session_id: str
    action: str  # "signin-link", "refused", "no-tool-call"
    sandbox_name: None = None
    reused: bool = False


class RoutingLauncher:
    """The env-U watcher's launcher: a ticket launches as the user; anything else gets the sign-in link.

    Quacks like :class:`DatabricksSandboxLauncher` for the dispatcher (environment id and key, base URL, prefix).
    Sweeps are off for the watcher: user sandboxes are owned by each user, not by the App.
    """

    def __init__(
        self,
        *,
        template: DatabricksSandboxLauncher,
        launcher_for: Callable[[WorkspaceClient], DatabricksSandboxLauncher],
        tickets: Tickets,
        store: Store,
        sessions_api: Any,
        agents: tuple[str, ...],
        app_url: str | None,
        work_api: Any | None = None,
        session_client_for: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        item_max_s: float = LINK_ITEM_MAX_S,
        poll_s: float = LINK_POLL_S,
    ) -> None:
        self.template = template
        self.environment_id = template.environment_id
        self.environment_key = template.environment_key
        self.anthropic_base_url = template.anthropic_base_url
        self.sandbox_id_prefix = template.sandbox_id_prefix
        self.workspace_client = template.workspace_client
        self.retention_days = template.retention_days
        self.lifecycle = template.lifecycle
        self.mcp_servers = template.mcp_servers
        self._launcher_for = launcher_for
        self.tickets = tickets
        self.store = store
        self.sessions_api = sessions_api
        self.agents = agents
        self.app_url = app_url.rstrip("/") if app_url else None
        self._work_api = work_api
        self._session_client_for = session_client_for or self._session_client
        self._clock, self._sleep = clock, sleep
        self._item_max_s, self._poll_s = item_max_s, poll_s

    def __repr__(self) -> str:
        return (
            f"RoutingLauncher(environment_id={self.environment_id!r}, tickets={len(self.tickets)})"
        )

    # The dispatcher calls this in a worker thread.
    def launch(self, work: Any) -> Any:
        session_id = work.data.id
        ticket = self.tickets.get(session_id)
        binding = self.store.binding(session_id)
        if (
            ticket is not None
            and binding is not None
            and binding.principal_id == ticket.principal_id
        ):
            try:
                result = self._launcher_for(ticket.client).launch(work)
            except Exception as e:
                self._audit(binding.principal_id, session_id, "launch-failed", type(e).__name__)
                raise
            self._audit(binding.principal_id, session_id, "launch", "")
            log.info("launched session=%s as its user", session_id)
            return result
        return self._answer_with_link(work, binding)

    # ------------------------------------------------------------ the sign-in link

    def _session_client(self, sessions_token: str) -> Any:
        # Only the self-hosted runner's session credential may post tool results; it is scoped to this session.
        return anthropic.Anthropic(auth_token=sessions_token, base_url=self.anthropic_base_url)

    def _answer_with_link(self, work: Any, binding: Binding | None) -> Deferred:
        session_id = work.data.id
        owner = binding.principal_id if binding else ""
        try:
            sessions_token = sessions_token_from_secret(getattr(work, "secret", None))
            if not sessions_token:
                self._audit(owner, session_id, "refused", "no sessions token")
                log.warning("session=%s: work item has no sessions token; released", session_id)
                return Deferred(session_id, "refused")
            runner = self._session_client_for(sessions_token)
            session = self.sessions_api.beta.sessions.retrieve(session_id)
            reason = refusal(session, environment_id=self.environment_id, agents=self.agents)
            if reason:
                message = f"This session can't run on Databricks: {reason}."
                action = "refused"
            elif not self.app_url:
                message = "Databricks sign-in is needed, but the App's URL is not configured. Ask the App's admin."
                action = "refused"
                reason = "no app url"
            else:
                code = secrets.token_urlsafe(24)
                self.store.add_code(code_hash(code), session_id, CODE_TTL_S)
                minutes = int(CODE_TTL_S // 60)
                message = (
                    "Databricks sign-in needed. Ask the user to open this link, sign in, confirm, and then reply "
                    f"'done': {self.app_url}/connect?code={code} . The link works once and expires in {minutes} "
                    "minutes. Don't call tools again until the user replies."
                )
                action = "signin-link"
            answered = self._answer_tool_calls(runner, session_id, message)
            if not answered:
                action = "no-tool-call"
            self._audit(owner, session_id, action, reason or "")
            log.info("session=%s: %s (answered %d tool call(s))", session_id, action, answered)
            return Deferred(session_id, action)
        finally:
            self._release(work)

    def _answer_tool_calls(self, runner: Any, session_id: str, message: str) -> int:
        """Answer every pending tool call with ``message`` until the turn ends (or the time limit). Returns the
        number answered. A turn that ends without a tool call needs nothing."""
        deadline = self._clock() + self._item_max_s
        answered: set[str] = set()
        while True:
            pending, idle_reason, terminated = self._pending(runner, session_id)
            for ev in pending:
                if ev.id in answered:
                    continue
                runner.beta.sessions.events.send(session_id, events=[_result_event(ev, message)])
                answered.add(ev.id)
            if terminated or (not pending and idle_reason not in (None, "requires_action")):
                return len(answered)
            if self._clock() >= deadline:
                return len(answered)
            self._sleep(self._poll_s)

    def _pending(self, runner: Any, session_id: str) -> tuple[list[Any], str | None, bool]:
        """Unanswered tool calls, the latest idle stop reason (None while running) and whether it terminated."""
        calls: dict[str, Any] = {}
        done: set[str] = set()
        idle_reason: str | None = None
        terminated = False
        for ev in runner.beta.sessions.events.list(
            session_id, order="asc", types=list(WATCH_EVENT_TYPES), limit=100
        ):
            etype = getattr(ev, "type", "")
            if etype in TOOL_USE_TYPES:
                calls[ev.id] = ev
            elif etype == "user.tool_result":
                done.add(ev.tool_use_id)
            elif etype == "user.custom_tool_result":
                done.add(ev.custom_tool_use_id)
            elif etype == "session.status_idle":
                idle_reason = getattr(getattr(ev, "stop_reason", None), "type", None) or "end_turn"
            elif etype == "session.status_running":
                idle_reason = None
            elif etype == "session.status_terminated":
                terminated = True
        return [ev for i, ev in calls.items() if i not in done], idle_reason, terminated

    def _release(self, work: Any) -> None:
        """Stop the item, so the session's next need for the environment queues a fresh one."""
        try:
            api = self._work_api or anthropic.Anthropic(
                auth_token=self.environment_key, base_url=self.anthropic_base_url
            )
            api.beta.environments.work.stop(work.id, environment_id=self.environment_id, force=True)
        except Exception as e:  # noqa: BLE001 - best effort; the lease expires anyway
            log.warning("could not release work=%s: %s", work.id, type(e).__name__)

    def _audit(self, principal_id: str, session_id: str, action: str, detail: str) -> None:
        try:
            self.store.audit(principal_id, session_id, action, detail)
        except Exception as e:  # noqa: BLE001 - auditing must not fail the item
            log.warning("audit failed for session=%s: %s", session_id, type(e).__name__)


def _result_event(ev: Any, text: str) -> dict[str, Any]:
    content = [{"type": "text", "text": text}]
    if getattr(ev, "type", "") == "agent.custom_tool_use":
        return {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": ev.id,
            "is_error": True,
            "content": content,
        }
    return {"type": "user.tool_result", "tool_use_id": ev.id, "is_error": True, "content": content}


def app_url_from_environment(env: Any, workspace_client: WorkspaceClient) -> str | None:
    """The App's public URL for sign-in links: ``CMA_APP_URL``, ``DATABRICKS_APP_URL``, or the App's own record."""
    for name in ("CMA_APP_URL", "DATABRICKS_APP_URL"):
        if env.get(name):
            return env[name]
    app_name = env.get("DATABRICKS_APP_NAME")
    if app_name:
        try:
            return workspace_client.apps.get(app_name).url
        except Exception as e:  # noqa: BLE001
            log.warning("could not read the App's URL: %s", type(e).__name__)
    return None
