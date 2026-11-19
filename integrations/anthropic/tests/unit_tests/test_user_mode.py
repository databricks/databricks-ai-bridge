"""Offline tests for user mode (M2) in ``app/``: the sign-in routes, the env-U watcher's routing launcher,
configuration, identity and the store.

No Databricks, Anthropic or Lakebase calls: identities, the Anthropic client, the store and the poller are fakes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
import service as svc  # noqa: E402
import signin  # noqa: E402
import user_mode as um  # noqa: E402
import user_store as us  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import SECURITY_HEADERS, create_app  # noqa: E402
from databricks_anthropic.sandbox.dispatcher import DispatcherStatus  # noqa: E402

SERVICE_ENV = {
    "ANTHROPIC_ENVIRONMENT_ID": "env_S",
    "ANTHROPIC_ENVIRONMENT_KEY": "sk-env-S-SECRET",
    "CMA_SANDBOX_ID_KEY": "hmac-SECRET",
}
USER_ENV = {
    "CMA_USER_ENVIRONMENT_ID": "env_U",
    "ANTHROPIC_USER_ENVIRONMENT_KEY": "sk-env-U-SECRET",
    "CMA_ANTHROPIC_API_KEY": "sk-ant-SECRET",
    "CMA_USER_AGENTS": "agent_one,agent_two",
}
AGENT_ONE, AGENT_TWO = "agent_one", "agent_two"
ALICE = um.Caller(principal_id="1001", user_name="alice@example.com")
BOB = um.Caller(principal_id="1002", user_name="bob@example.com")
TOKENS = {"tok-alice-SECRET": ALICE, "tok-bob-SECRET": BOB}
HOST = "https://app.example"
GOOD_ORIGIN = {"origin": HOST, "x-cma-request": "1"}


def config(**overrides) -> um.UserConfig:
    base: dict[str, Any] = dict(
        environment_id="env_U",
        environment_key="sk-env-U-SECRET",
        anthropic_api_key="sk-ant-SECRET",
        agents=(AGENT_ONE, AGENT_TWO),
        max_sessions_per_user=20,
    )
    return um.UserConfig(**{**base, **overrides})


def service_launcher():
    return svc.build_launcher(SERVICE_ENV, workspace_client=MagicMock())


class IdleDispatcher:
    """Stands in for the service poller: runs until cancelled."""

    def __init__(self, launcher, *, max_concurrent_launches):
        self.status = DispatcherStatus()

    async def run(self):
        self.status.polling = True
        try:
            await asyncio.Event().wait()
        finally:
            self.status.polling = False


def idle_service():
    return svc.DispatcherService(service_launcher(), make_dispatcher=cast(Any, IdleDispatcher))


class FakeIdentity:
    def __init__(self):
        self.fresh_calls = 0

    def resolve(self, token, *, fresh=False):
        self.fresh_calls += bool(fresh)
        if token not in TOKENS:
            raise PermissionError("token rejected by the workspace: " + token[:4])
        return TOKENS[token], MagicMock(name="user-client")


class FakeStore(us.MemoryStore):
    """A MemoryStore that can be switched off to simulate Lakebase being down."""

    down = False

    def _check(self):
        if self.down:
            raise us.StoreUnavailable("Lakebase down")

    def bind(self, binding):
        self._check()
        return super().bind(binding)

    def binding(self, session_id):
        self._check()
        return super().binding(session_id)

    def bindings_for(self, principal_id):
        self._check()
        return super().bindings_for(principal_id)

    def count_for(self, principal_id):
        self._check()
        return super().count_for(principal_id)

    def audit(self, principal_id, session_id, action, detail=""):
        self._check()
        return super().audit(principal_id, session_id, action, detail)

    def peek_code(self, code_hash):
        self._check()
        return super().peek_code(code_hash)


class FakeAnthropic:
    """Only what the sign-in routes read: sessions.retrieve and agents.retrieve. Anything else would fail."""

    def __init__(self):
        self.sessions: dict[str, SimpleNamespace] = {}  # console-started sessions, by id
        self.beta = SimpleNamespace(
            agents=SimpleNamespace(retrieve=self._agent),
            sessions=SimpleNamespace(retrieve=self._retrieve),
        )

    def _retrieve(self, session_id):
        return self.sessions.get(session_id) or console_session(session_id)

    def _agent(self, agent_id):
        if agent_id == AGENT_ONE:
            return SimpleNamespace(name="demo_one")
        raise RuntimeError("lookup failed")  # the page falls back to the id


def console_session(
    session_id, agent=AGENT_ONE, environment_id="env_U", resources=(), vault_ids=()
):
    return SimpleNamespace(
        id=session_id,
        agent=SimpleNamespace(id=agent),
        environment_id=environment_id,
        resources=list(resources),
        vault_ids=list(vault_ids),
    )


class RecordingTickets(signin.Tickets):
    """Tickets that also record each put (the env-U watcher isn't running in route tests)."""

    def __init__(self):
        super().__init__()
        self.put_calls: list = []

    def put(self, session_id, principal_id, client, *, ttl_s):
        self.put_calls.append((session_id, principal_id, client, ttl_s))
        return super().put(session_id, principal_id, client, ttl_s=ttl_s)


def make_user(*, cfg=None, store=None, anth=None, tickets=None, app_url=HOST):
    return um.UserMode(
        config=cfg or config(),
        identity=cast(Any, FakeIdentity()),
        store=store or FakeStore(),
        anthropic=anth or FakeAnthropic(),
        service_launcher=service_launcher(),
        tickets=tickets if tickets is not None else RecordingTickets(),
        app_url=app_url,
    )


def make_client(*, cfg=None, store=None, anth=None, tickets=None, user=None):
    user = user or make_user(cfg=cfg, store=store, anth=anth, tickets=tickets)
    cast(Any, user).start = lambda: None  # no env-U watcher in route tests
    app = create_app(make_service=idle_service, make_user_mode=lambda service: user)
    return TestClient(app, base_url=HOST)


@pytest.fixture
def user_env(monkeypatch):
    monkeypatch.setenv("CMA_USER_MODE", "1")


def auth(token="tok-alice-SECRET", **extra):
    return {"x-forwarded-access-token": token, **GOOD_ORIGIN, **extra}


CODE = "c0de-" + "x" * 30


def with_code(store, session_id="sesn_console", ttl_s=900):
    store.add_code(signin.code_hash(CODE), session_id, ttl_s)
    return session_id


# ---------------------------------------------------------------- routes present or absent


def test_user_routes_absent_without_user_mode(monkeypatch):
    monkeypatch.delenv("CMA_USER_MODE", raising=False)
    with TestClient(create_app(make_service=idle_service), base_url=HOST) as client:
        assert client.get("/connect").status_code == 404
        assert client.get("/connect.js").status_code == 404
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).status_code == 404
        )
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code in (
            404,
            405,
        )


def test_user_mode_off_when_flag_is_not_exactly_one(monkeypatch):
    monkeypatch.setenv("CMA_USER_MODE", "true")
    with TestClient(create_app(make_service=idle_service), base_url=HOST) as client:
        assert client.get("/connect").status_code == 404


def test_only_the_sign_in_routes_exist(user_env):
    # The session API and the chat page are not part of the App.
    with make_client() as client:
        for path in (
            "/chat",
            "/chat.js",
            "/api/agents",
            "/api/sessions",
            "/api/sessions/sesn_1/events",
        ):
            assert client.get(path, headers=auth()).status_code == 404, path
        assert client.post("/api/sessions", json={}, headers=auth()).status_code in (404, 405)


# ---------------------------------------------------------------- request hygiene


def test_state_changes_need_the_header_and_same_origin(user_env):
    store, tickets = FakeStore(), RecordingTickets()
    sid = with_code(store)
    with make_client(store=store, tickets=tickets) as client:
        no_header = {"x-forwarded-access-token": "tok-alice-SECRET", "origin": HOST}
        assert (
            client.post("/api/connect", json={"code": CODE}, headers=no_header).status_code == 403
        )
        cross = auth(origin="https://evil.example")
        assert client.post("/api/connect", json={"code": CODE}, headers=cross).status_code == 403
        assert (
            client.post(
                "/api/connect", json={"code": CODE}, headers=auth(origin="null")
            ).status_code
            == 403
        )
        no_origin = {"x-forwarded-access-token": "tok-alice-SECRET", "x-cma-request": "1"}
        assert (
            client.post("/api/connect", json={"code": CODE}, headers=no_origin).status_code == 403
        )
    assert store.binding(sid) is None and tickets.put_calls == []


def test_extra_fields_and_bad_codes_are_rejected(user_env):
    with make_client() as client:
        assert (
            client.post(
                "/api/connect", json={"code": CODE, "session_id": "x"}, headers=auth()
            ).status_code
            == 422
        )
        assert (
            client.post("/api/connect", json={"code": "short"}, headers=auth()).status_code == 422
        )
        assert (
            client.post("/api/connect/info", json={"code": "short"}, headers=auth()).status_code
            == 422
        )


def test_missing_or_rejected_token_is_401(user_env):
    with make_client() as client:
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=GOOD_ORIGIN).status_code
            == 401
        )
        nobody = auth(token="tok-nobody")
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=nobody).status_code == 401
        )


# ---------------------------------------------------------------- failure modes and secrecy


def test_store_unavailable_is_503(user_env):
    store = FakeStore()
    with_code(store)
    with make_client(store=store) as client:
        store.down = True
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).status_code == 503
        )
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code == 503


def test_token_and_code_never_appear_in_any_response(user_env):
    store = FakeStore()
    with_code(store)
    responses = []
    with make_client(store=store) as client:
        responses.append(client.get("/connect", headers=auth()))
        responses.append(client.get("/connect.js", headers=auth()))
        responses.append(client.post("/api/connect/info", json={"code": CODE}, headers=auth()))
        responses.append(client.post("/api/connect", json={"code": CODE}, headers=auth()))
        responses.append(
            client.post(
                "/api/connect/info", json={"code": CODE}, headers=auth(token="tok-bob-SECRET")
            )
        )
        bad = {"x-forwarded-access-token": "nope-SECRET", **GOOD_ORIGIN}
        responses.append(client.post("/api/connect/info", json={"code": CODE}, headers=bad))
    secrets = [
        "tok-alice-SECRET",
        "tok-bob-SECRET",
        "nope-SECRET",
        "sk-ant-SECRET",
        "sk-env-U-SECRET",
        CODE,
    ]
    for r in responses:
        for secret in secrets:
            assert secret not in r.text
            assert all(secret not in v for v in r.headers.values())


# ---------------------------------------------------------------- the /connect page


def test_connect_script_never_uses_innerhtml():
    source = (Path(__file__).resolve().parents[2] / "app" / "static" / "connect.js").read_text()
    assert not re.search(r"\.innerHTML\s*=|insertAdjacentHTML|document\.write|\beval\(", source)


def test_connect_page_has_the_strict_csp(user_env):
    with make_client() as client:
        page, script = client.get("/connect"), client.get("/connect.js")
        api = client.post("/api/connect/info", json={"code": CODE}, headers=auth())
    assert page.status_code == 200 and "/connect.js" in page.text and "your" in page.text
    assert page.headers["content-security-policy"] == um.PAGE_CSP
    assert "script-src 'self'" in um.PAGE_CSP and "frame-ancestors 'none'" in um.PAGE_CSP
    assert script.headers["content-security-policy"] == um.PAGE_CSP
    # The middleware keeps the other headers, and never overwrites a route's own policy.
    for name, value in SECURITY_HEADERS.items():
        if name != "Content-Security-Policy":
            assert api.headers[name] == value
    assert api.headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]


# ---------------------------------------------------------------- the sign-in link: routes


def test_connect_binds_a_console_session_to_whoever_confirms_and_works_once(user_env):
    store, tickets = FakeStore(), RecordingTickets()
    sid = with_code(store)
    with make_client(store=store, tickets=tickets) as client:
        info = client.post("/api/connect/info", json={"code": CODE}, headers=auth())
        assert info.status_code == 200, info.text
        assert info.json() == {
            "session_id": sid,
            "agent": "demo_one",
            "signed_in_as": ALICE.user_name,
            "already_yours": False,
            "problem": None,
        }
        assert store.binding(sid) is None  # looking consumes nothing
        r = client.post("/api/connect", json={"code": CODE}, headers=auth())
        assert r.status_code == 200, r.text
        assert "reply 'done'" in r.json()["note"]
        again = client.post("/api/connect", json={"code": CODE}, headers=auth())
        assert again.status_code == 404
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).status_code == 404
        )
    binding = store.binding(sid)
    assert (binding.principal_id, binding.agent_id) == (ALICE.principal_id, AGENT_ONE)
    ((ticket_sid, principal, _c, ttl),) = tickets.put_calls
    assert (ticket_sid, principal) == (
        sid,
        ALICE.principal_id,
    ) and 0 < ttl <= signin.LINK_TICKET_MAX_S
    assert (ALICE.principal_id, sid, "connect", AGENT_ONE) in store.audit_rows
    assert CODE not in str(store.audit_rows)


def test_connect_never_takes_over_another_users_session(user_env):
    store, tickets = FakeStore(), RecordingTickets()
    sid = with_code(store)
    store.bind(us.Binding(sid, BOB.principal_id, BOB.user_name, AGENT_ONE))
    with make_client(store=store, tickets=tickets) as client:
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).status_code == 404
        )
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code == 404
        # The owner can still use it.
        assert (
            client.post(
                "/api/connect", json={"code": CODE}, headers=auth(token="tok-bob-SECRET")
            ).status_code
            == 200
        )
    assert store.binding(sid).principal_id == BOB.principal_id
    assert [t[1] for t in tickets.put_calls] == [BOB.principal_id]
    assert (ALICE.principal_id, sid, "connect-refused", "bound to another user") in store.audit_rows


def test_owner_reconnects_a_bound_session_without_a_new_binding(user_env):
    store, tickets = FakeStore(), RecordingTickets()
    sid = with_code(store)
    store.bind(us.Binding(sid, ALICE.principal_id, ALICE.user_name, AGENT_ONE))
    with make_client(store=store, tickets=tickets) as client:
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).json()[
                "already_yours"
            ]
            is True
        )
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code == 200
    assert store.count_for(ALICE.principal_id) == 1 and len(tickets.put_calls) == 1


@pytest.mark.parametrize(
    "session, problem",
    [
        (console_session("sesn_console", agent="agent_other"), "agent is not enabled"),
        (
            console_session("sesn_console", environment_id="env_S"),
            "not on the user-mode environment",
        ),
        (
            console_session("sesn_console", resources=[SimpleNamespace(type="memory_store")]),
            "memory stores",
        ),
        (console_session("sesn_console", vault_ids=["vlt_1"]), "vaults"),
    ],
)
def test_connect_refuses_sessions_user_mode_cannot_run(user_env, session, problem):
    store, anth, tickets = FakeStore(), FakeAnthropic(), RecordingTickets()
    anth.sessions["sesn_console"] = session
    sid = with_code(store)
    with make_client(store=store, anth=anth, tickets=tickets) as client:
        assert (
            problem
            in client.post("/api/connect/info", json={"code": CODE}, headers=auth()).json()[
                "problem"
            ]
        )
        r = client.post("/api/connect", json={"code": CODE}, headers=auth())
    assert r.status_code == 403 and problem in r.json()["detail"]
    assert store.binding(sid) is None and tickets.put_calls == []
    assert store.peek_code(signin.code_hash(CODE)) == sid  # a refusal doesn't burn the code


def test_connect_rejects_expired_codes_missing_headers_and_the_session_cap(user_env):
    now = [1000.0]
    store = FakeStore(clock=lambda: now[0])
    with_code(store, ttl_s=60)
    with make_client(store=store, cfg=config(max_sessions_per_user=0)) as client:
        no_header = {"x-forwarded-access-token": "tok-alice-SECRET", "origin": HOST}
        assert (
            client.post("/api/connect", json={"code": CODE}, headers=no_header).status_code == 403
        )
        assert (
            client.post("/api/connect", json={"code": CODE, "x": 1}, headers=auth()).status_code
            == 422
        )
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code == 429
        now[0] += 61
        assert client.post("/api/connect", json={"code": CODE}, headers=auth()).status_code == 404
        assert (
            client.post("/api/connect/info", json={"code": "y" * 40}, headers=auth()).status_code
            == 404
        )


# ---------------------------------------------------------------- the env-U watcher's routing launcher


def ev(etype, eid, **kw):
    return SimpleNamespace(type=etype, id=eid, **kw)


def idle(eid, reason):
    return ev("session.status_idle", eid, stop_reason=SimpleNamespace(type=reason))


class ScriptedSessions:
    """sessions.retrieve/events for the routing launcher: each events.list call returns the next page."""

    def __init__(self, pages, session=None):
        self.pages, self.sent, self.tokens = list(pages), [], []
        self.session = session or console_session("sesn_c")
        self.beta = SimpleNamespace(
            sessions=SimpleNamespace(
                retrieve=lambda sid: self.session,
                events=SimpleNamespace(list=self._list, send=self._send),
            )
        )

    def _list(self, sid, **kw):
        return self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]

    def _send(self, sid, *, events):
        self.sent.extend(events)


class InlineExecutor:
    """Runs a submitted call at once, so tests see the link answered when launch() returns."""

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001 - surfaced through the future, as a real pool would
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def routing(store, sessions, *, tickets=None, launches=None, app_url=HOST, fail=None):
    launched = launches if launches is not None else []
    principals: list[str] = []
    work_api = MagicMock()

    def launcher_for(client, principal_id):
        def launch(work):
            if fail:
                raise fail
            launched.append((work.data.id, client))
            principals.append(principal_id)
            return SimpleNamespace(sandbox_name=f"sandboxes/cmausr-{work.data.id}")

        return SimpleNamespace(launch=launch)

    template = um.user_launcher(service_launcher(), config(), MagicMock())
    rl = signin.RoutingLauncher(
        template=template,
        launcher_for=launcher_for,
        tickets=tickets or signin.Tickets(),
        store=store,
        sessions_api=sessions,
        agents=(AGENT_ONE, AGENT_TWO),
        app_url=app_url,
        work_api=work_api,
        session_client_for=lambda token: (sessions.tokens.append(token), sessions)[1],
        sleep=lambda s: None,
        poll_s=0,
        link_executor=cast(Any, InlineExecutor()),
    )
    return rl, launched, work_api


def item(session_id="sesn_c", secret=None):
    import base64
    import json as _json

    if secret is None:
        secret = base64.urlsafe_b64encode(
            _json.dumps({"sessions_token": "tok-sess-SECRET"}).encode()
        ).decode()
    return SimpleNamespace(
        id=session_id, secret=secret, data=SimpleNamespace(id=session_id, type="session")
    )


def test_watcher_launches_as_the_user_holding_the_ticket():
    store, tickets = us.MemoryStore(), signin.Tickets()
    store.bind(us.Binding("sesn_c", ALICE.principal_id, ALICE.user_name, AGENT_ONE))
    alice_client = MagicMock(name="alice")
    tickets.put("sesn_c", ALICE.principal_id, alice_client, ttl_s=60)
    rl, launched, work_api = routing(store, ScriptedSessions([[]]), tickets=tickets)
    result = rl.launch(item())
    assert (
        launched == [("sesn_c", alice_client)] and result.sandbox_name == "sandboxes/cmausr-sesn_c"
    )
    assert store.audit_rows[-1] == (ALICE.principal_id, "sesn_c", "launch", "")
    work_api.beta.environments.work.stop.assert_not_called()  # the worker owns the item now
    # The watcher quacks like the template launcher on env U.
    assert (rl.environment_id, rl.sandbox_id_prefix) == ("env_U", "cmausr-")
    assert "SECRET" not in repr(rl)


def test_watcher_audits_a_failed_launch_and_lets_the_dispatcher_release_the_item():
    store, tickets = us.MemoryStore(), signin.Tickets()
    store.bind(us.Binding("sesn_c", ALICE.principal_id, ALICE.user_name, AGENT_ONE))
    tickets.put("sesn_c", ALICE.principal_id, MagicMock(), ttl_s=60)
    rl, _, _ = routing(store, ScriptedSessions([[]]), tickets=tickets, fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):  # the dispatcher logs it and force-stops the item
        rl.launch(item())
    assert store.audit_rows[-1][2:] == ("launch-failed", "RuntimeError")


def test_watcher_answers_tool_calls_with_a_one_time_link_when_no_ticket_waits():
    store = us.MemoryStore()
    pages = [
        [
            ev("agent.tool_use", "t0"),
            ev("user.tool_result", "r0", tool_use_id="t0"),  # already answered: left alone
            ev("agent.tool_use", "t1"),
            ev("agent.custom_tool_use", "t2"),
            idle("i1", "requires_action"),
        ],
        [
            ev("agent.tool_use", "t1"),
            ev("agent.custom_tool_use", "t2"),
            idle("i1", "requires_action"),
        ],
        [
            ev("agent.tool_use", "t1"),
            ev("user.tool_result", "r1", tool_use_id="t1"),
            ev("agent.custom_tool_use", "t2"),
            ev("user.custom_tool_result", "r2", custom_tool_use_id="t2"),
            idle("i2", "end_turn"),
        ],
    ]
    sessions = ScriptedSessions(pages)
    rl, launched, work_api = routing(store, sessions)
    result = rl.launch(item())
    assert launched == [] and result.action == "signin-link"
    assert [e["type"] for e in sessions.sent] == ["user.tool_result", "user.custom_tool_result"]
    assert (
        sessions.sent[0]["tool_use_id"] == "t1" and sessions.sent[1]["custom_tool_use_id"] == "t2"
    )
    assert all(e["is_error"] for e in sessions.sent)
    text = sessions.sent[0]["content"][0]["text"]
    found = re.search(r"/connect\?code=([A-Za-z0-9]+)", text)
    assert found
    code = found.group(1)
    assert text.count(f"{HOST}/connect?code=") == 1 and "reply 'done'" in text
    assert store.peek_code(signin.code_hash(code)) == "sesn_c"  # stored hashed, redeemable
    assert code not in str(store.audit_rows) and store.audit_rows[-1][2] == "signin-link"
    assert sessions.tokens == [
        "tok-sess-SECRET"
    ]  # results go out with the session's own runner credential
    work_api.beta.environments.work.stop.assert_called_once_with(
        "sesn_c", environment_id="env_U", force=True
    )


def test_a_ticket_for_another_user_never_launches():
    store, tickets = us.MemoryStore(), signin.Tickets()
    store.bind(us.Binding("sesn_c", BOB.principal_id, BOB.user_name, AGENT_ONE))
    tickets.put("sesn_c", ALICE.principal_id, MagicMock(), ttl_s=60)
    sessions = ScriptedSessions([[idle("i1", "end_turn")]])
    rl, launched, _ = routing(store, sessions, tickets=tickets)
    assert rl.launch(item()).action == "no-tool-call" and launched == []


def test_expired_tickets_are_ignored():
    now = [0.0]
    tickets = signin.Tickets(clock=lambda: now[0])
    tickets.put("sesn_c", ALICE.principal_id, MagicMock(), ttl_s=10)
    now[0] = 11
    assert tickets.get("sesn_c") is None and len(tickets) == 0


@pytest.mark.parametrize(
    "session, app_url, phrase",
    [
        (console_session("sesn_c", agent="agent_other"), HOST, "agent is not enabled"),
        (
            console_session("sesn_c", resources=[SimpleNamespace(type="memory_store")]),
            HOST,
            "memory stores",
        ),
        (console_session("sesn_c"), None, "URL is not configured"),
    ],
)
def test_watcher_refuses_without_a_link_when_the_session_cannot_run(session, app_url, phrase):
    store = us.MemoryStore()
    sessions = ScriptedSessions(
        [
            [ev("agent.tool_use", "t1"), idle("i", "requires_action")],
            [
                ev("agent.tool_use", "t1"),
                ev("user.tool_result", "r", tool_use_id="t1"),
                idle("i2", "end_turn"),
            ],
        ],
        session=session,
    )
    rl, launched, work_api = routing(store, sessions, app_url=app_url)
    assert rl.launch(item()).action == "refused" and launched == []
    text = sessions.sent[0]["content"][0]["text"]
    assert phrase in text and "/connect" not in text
    work_api.beta.environments.work.stop.assert_called_once()


def test_watcher_gives_up_answering_after_its_time_limit():
    store = us.MemoryStore()
    sessions = ScriptedSessions([[ev("agent.tool_use", "t1")]])  # never ends (still running)
    rl, _, work_api = routing(store, sessions)
    clock = [0.0]
    rl._clock = lambda: clock[0]
    rl._sleep = lambda s: clock.__setitem__(0, clock[0] + 100)
    assert rl.launch(item()).action == "signin-link"
    assert len(sessions.sent) == 1  # answered once, not re-answered every poll
    work_api.beta.environments.work.stop.assert_called_once()


def jwt(exp):
    import base64
    import json as _json

    body = base64.urlsafe_b64encode(_json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"h.{body}.s"


def test_link_tickets_expire_before_the_users_token():
    import time as _time

    assert signin.link_ticket_ttl(jwt(_time.time() + 3600)) == signin.LINK_TICKET_MAX_S
    assert 0 < signin.link_ticket_ttl(jwt(_time.time() + 600)) <= 600 - signin.TOKEN_MARGIN_S + 1
    assert signin.link_ticket_ttl(jwt(_time.time() + 60)) == 0
    assert signin.link_ticket_ttl("not-a-jwt") == 15 * 60


def test_memory_store_codes_are_single_use_and_expire():
    now = [0.0]
    store = us.MemoryStore(clock=lambda: now[0])
    store.add_code("h1", "sesn_1", 60)
    assert store.peek_code("h1") == "sesn_1" and store.peek_code("h1") == "sesn_1"
    assert store.use_code("h1", "1001") == "sesn_1" and store.use_code("h1", "1002") is None
    store.add_code("h2", "sesn_2", 60)
    now[0] = 61
    assert store.peek_code("h2") is None and store.use_code("h2", "1001") is None


def test_lakebase_codes_use_an_insert_for_single_use():
    log: list = []
    store, _ = lakebase(lambda **kw: FakeConnection(log))
    store.add_code("h1", "sesn_1", 900)
    store.peek_code("h1")
    store.use_code("h1", "1001")
    statements = [sql for sql, _ in log if "connect_" in sql and not sql.startswith("CREATE")]
    assert statements[0].startswith("INSERT INTO cma_dispatcher.connect_codes")
    assert all(not re.match(r"(UPDATE|DELETE)\b", sql) for sql in statements)


# ---------------------------------------------------------------- configuration and launcher


def test_missing_user_configuration_names_keys_not_values():
    env = {k: v for k, v in USER_ENV.items() if k != "CMA_ANTHROPIC_API_KEY"}
    with pytest.raises(um.ConfigError) as e:
        um.UserConfig.from_environment(env, service_environment_id="env_S")
    assert "CMA_ANTHROPIC_API_KEY" in str(e.value)
    assert "SECRET" not in str(e.value)


def test_user_environment_must_differ_from_the_service_environment():
    with pytest.raises(um.ConfigError, match="must differ"):
        um.UserConfig.from_environment(USER_ENV, service_environment_id="env_U")


def test_user_config_parses_agents_and_the_session_cap():
    cfg = um.UserConfig.from_environment(
        {**USER_ENV, "CMA_MAX_SESSIONS_PER_USER": "5"}, service_environment_id="env_S"
    )
    assert cfg.agents == ("agent_one", "agent_two") and cfg.max_sessions_per_user == 5
    assert "sk-ant" not in repr(cfg)


def test_user_launcher_retargets_environment_and_identity_and_keeps_the_service_launcher():
    base = service_launcher()
    client = MagicMock(name="user-client")
    lc = um.user_launcher(base, config(), client)
    assert lc.environment_id == "env_U" and lc.environment_key == "sk-env-U-SECRET"
    assert lc.workspace_client is client and lc.sandbox_id_prefix == "cmausr-"
    assert lc.sandbox_id_key == base.sandbox_id_key and lc.lifecycle == "ephemeral"
    assert base.environment_id == "env_S" and base.sandbox_id_prefix == "cmaapp-"


def test_replace_reruns_launcher_validation():
    base = service_launcher()
    with pytest.raises(ValueError, match="sandbox_id_prefix"):
        dataclasses.replace(base, sandbox_id_prefix="NOT VALID")
    with pytest.raises(ValueError, match="https"):
        dataclasses.replace(base, anthropic_base_url="http://proxy")


def test_user_launcher_names_its_sandboxes_differently_from_the_service():
    base = service_launcher()
    lc = um.user_launcher(base, config(), MagicMock())
    assert lc.sandbox_id_for("sesn_1").startswith("cmausr-")
    assert lc.sandbox_id_for("sesn_1") != base.sandbox_id_for("sesn_1")


# ---------------------------------------------------------------- identity


def test_identity_caches_principals_by_token_digest_only():
    calls = []

    def make(host, token):
        client = MagicMock()

        def me():
            calls.append(token)
            return SimpleNamespace(id="1001", user_name="alice@example.com")

        client.current_user.me = me
        return client

    now = [0.0]
    identity = um.Identity("https://h", ttl_s=60, clock=lambda: now[0], client_factory=make)
    caller, _ = identity.resolve("tok-alice-SECRET")
    again, _ = identity.resolve("tok-alice-SECRET")
    assert caller == again == um.Caller("1001", "alice@example.com")
    assert len(calls) == 1
    now[0] = 61
    identity.resolve("tok-alice-SECRET")
    assert len(calls) == 2
    assert "SECRET" not in str(identity._cache)  # only digests and principals are kept


def test_identity_refuses_an_identity_without_an_id():
    def make(host, token):
        client = MagicMock()
        client.current_user.me.return_value = SimpleNamespace(id="", user_name="")
        return client

    with pytest.raises(PermissionError):
        um.Identity("https://h", client_factory=make).resolve("tok")


# ---------------------------------------------------------------- store


def test_memory_store_keeps_one_binding_per_session_and_lists_per_principal():
    store = us.MemoryStore()
    store.bind(us.Binding("sesn_1", "1001", "alice", AGENT_ONE))
    store.bind(us.Binding("sesn_2", "1002", "bob", AGENT_ONE))
    with pytest.raises(us.StoreUnavailable):
        store.bind(us.Binding("sesn_1", "1002", "bob", AGENT_ONE))
    assert [b.session_id for b in store.bindings_for("1001")] == ["sesn_1"]
    assert store.count_for("1002") == 1
    store.audit("1001", "sesn_1", "create", AGENT_ONE)
    assert store.audit_rows == [("1001", "sesn_1", "create", AGENT_ONE)]


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.fail:
            raise OSError("could not translate host name")
        self.log.append((" ".join(sql.split()), params))
        if sql.lstrip().upper().startswith("SELECT COUNT"):
            return FakeCursor([(3,)])
        return FakeCursor([])

    def close(self):
        pass


def lakebase(connect, w=None):
    w = w or MagicMock()
    w.postgres.generate_database_credential.return_value = SimpleNamespace(token="pw-SECRET")
    return us.LakebaseStore(
        host="h.example",
        port="5432",
        database="databricks_postgres",
        user="sp-client-id",
        sslmode="require",
        endpoint="projects/p/branches/b/endpoints/e",
        workspace_client=w,
        connect=connect,
    ), w


def test_lakebase_store_creates_its_schema_then_only_inserts_into_audit():
    log: list = []
    store, w = lakebase(lambda **kw: FakeConnection(log))
    store.bind(us.Binding("sesn_1", "1001", "alice", AGENT_ONE))
    store.audit("1001", "sesn_1", "create", AGENT_ONE)
    assert store.count_for("1001") == 3
    statements = [sql for sql, _ in log]
    assert statements[0].startswith("CREATE SCHEMA IF NOT EXISTS cma_dispatcher")
    assert any(s.startswith("INSERT INTO cma_dispatcher.audit") for s in statements)
    assert all(not re.match(r"(UPDATE|DELETE|TRUNCATE|DROP)\b", s) for s in statements)
    assert all("public" not in s for s in statements)
    # The password is fetched once for an hour, not per connection.
    assert w.postgres.generate_database_credential.call_count == 1


def test_lakebase_store_failures_do_not_leak_the_credential():
    def connect(**kwargs):
        raise OSError("pw-SECRET in a message")

    store, w = lakebase(connect)
    with pytest.raises(us.StoreUnavailable) as e:
        store.count_for("1001")
    assert "SECRET" not in str(e.value) and "OSError" in str(e.value)


def test_lakebase_store_drops_a_rejected_credential():
    attempts = []

    def connect(**kwargs):
        attempts.append(kwargs["password"])
        if len(attempts) == 1:
            raise OSError("auth failed")
        return FakeConnection([])

    store, w = lakebase(connect)
    with pytest.raises(us.StoreUnavailable):
        store.count_for("1001")
    store.count_for("1001")
    assert w.postgres.generate_database_credential.call_count == 2


def test_unattached_lakebase_answers_503_not_a_startup_failure(user_env, monkeypatch):
    for k in ("PGHOST", "PGDATABASE", "PGUSER", "LAKEBASE_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    user = um.user_mode_from_environment(
        SimpleNamespace(launcher=service_launcher()),
        env={**SERVICE_ENV, **USER_ENV},
    )
    assert isinstance(user.store, us.UnavailableStore)
    user.identity = cast(Any, FakeIdentity())
    cast(Any, user).start = lambda: None
    app = create_app(make_service=idle_service, make_user_mode=lambda service: user)
    with TestClient(app, base_url=HOST) as client:
        assert (
            client.post("/api/connect/info", json={"code": CODE}, headers=auth()).status_code == 503
        )
        assert client.get("/connect").status_code == 200  # the page itself doesn't need the store


def test_store_source_only_deletes_expired_sign_in_codes():
    """No UPDATE, TRUNCATE, DROP or ALTER anywhere; the only DELETEs are the two sign-in-code purges."""
    source = (Path(__file__).resolve().parents[2] / "app" / "user_store.py").read_text()
    assert not re.search(r"[\"'](\s*)(UPDATE|TRUNCATE|DROP|ALTER)\b", source)
    deletes = re.findall(r"DELETE FROM \{SCHEMA\}\.(\w+)", source)
    assert deletes == ["connect_codes", "connect_uses"]
    assert len(re.findall(r"[\"'](\s*)DELETE\b", source)) == 2


def test_watcher_releases_an_item_without_a_sessions_token():
    store = us.MemoryStore()
    sessions = ScriptedSessions([[ev("agent.tool_use", "t1")]])
    rl, _, work_api = routing(store, sessions)
    assert rl.launch(item(secret="")).action == "refused"
    assert sessions.sent == [] and sessions.tokens == []
    work_api.beta.environments.work.stop.assert_called_once()


def test_sign_in_link_answers_run_off_the_launch_slots():
    """A session waiting for a sign-in is answered on the link pool: launch() returns at once, so the dispatcher's
    launch slot is free for real sandbox launches."""
    import threading

    store = us.MemoryStore()
    gate, started = threading.Event(), threading.Event()
    rl, _, _ = routing(store, ScriptedSessions([[]]))
    rl._links = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)

    def slow_answer(work, binding):
        started.set()
        gate.wait(5)
        return signin.Deferred(work.data.id, "signin-link")

    setattr(rl, "_answer_with_link", slow_answer)  # noqa: B010 - replace the bound method for this test
    result = rl.launch(item("sesn_wait"))
    assert result.action == "answering" and result.launched is False  # returned without waiting
    assert started.wait(5)
    gate.set()
    rl.close()


def test_the_link_wait_is_capped_at_two_minutes():
    assert signin.LINK_ITEM_MAX_S == 120.0 and signin.LINK_WORKERS == 4


def test_confirm_always_rechecks_the_token_but_the_page_lookup_may_use_the_cache():
    """POST /api/connect (the state change) resolves the caller with fresh=True; the info lookup doesn't."""
    clock = [0.0]
    calls = []

    class Client:
        def __init__(self, token):
            me = SimpleNamespace(id="u1", user_name="a@x")
            self.current_user = SimpleNamespace(me=lambda: calls.append(token) or me)

    identity = um.Identity(
        HOST, ttl_s=60, clock=lambda: clock[0], client_factory=lambda host, token: Client(token)
    )
    identity.resolve("tok-1")
    identity.resolve("tok-1")
    assert len(calls) == 1  # cached for the page lookup
    identity.resolve("tok-1", fresh=True)
    assert len(calls) == 2  # the confirm step always asks Databricks


def test_lookups_are_post_only_and_codes_have_no_trailing_punctuation(user_env):
    """The link carries ?code= (the Apps sign-in redirect drops fragments); every API call sends it in the body."""
    assert "/connect?code=" in pathlib.Path(signin.__file__).read_text()
    script = (pathlib.Path(um.STATIC) / "connect.js").read_text()
    assert "window.location.search" in script and 'replaceState(null, "", "/connect")' in script
    assert '"/api/connect?code=' not in script
    codes = [signin.new_code() for _ in range(200)]
    assert all(re.fullmatch(r"[A-Za-z0-9]{32}", c) for c in codes) and len(set(codes)) == 200
    with make_client() as client:
        assert client.get(f"/api/connect?code={CODE}", headers=auth()).status_code in (404, 405)


def test_lakebase_purge_deletes_only_expired_codes_and_their_uses():
    log: list = []
    store, _ = lakebase(lambda **kw: FakeConnection(log))
    store.purge_expired_codes()
    deletes = [(sql, params) for sql, params in log if sql.startswith("DELETE")]
    assert [sql.split()[2] for sql, _ in deletes] == [
        "cma_dispatcher.connect_codes",
        "cma_dispatcher.connect_uses",
    ]
    assert "expires_at < now() - make_interval(secs => %s)" in deletes[0][0] and deletes[0][1] == (
        86400.0,
    )
    assert not any(("bindings" in sql or ".audit" in sql) for sql, _ in deletes)


def test_memory_store_purges_codes_a_day_after_expiry_and_keeps_live_ones():
    now = [0.0]
    store = us.MemoryStore(clock=lambda: now[0])
    store.add_code("old", "sesn_1", 60)
    store.use_code("old", "1001")
    store.add_code("live", "sesn_2", 60)
    now[0] = 60 + us.PURGE_GRACE_S - 1
    assert store.purge_expired_codes() == 0  # expired, but not a day ago yet
    store.add_code("new", "sesn_3", 900)
    now[0] = 61 + us.PURGE_GRACE_S
    assert (
        store.purge_expired_codes() == 2
    )  # "old" and "live" (both expired a day ago); "new" stays
    assert store.peek_code("new") == "sesn_3" and store.use_code("old", "1002") is None


def test_the_purge_never_raises():
    class Broken(us.MemoryStore):
        def purge_expired_codes(self, grace_s=us.PURGE_GRACE_S):
            raise OSError("lakebase down")

    user = um.UserMode(
        config=config(),
        identity=cast(Any, FakeIdentity()),
        store=Broken(),
        anthropic=MagicMock(),
        service_launcher=service_launcher(),
    )
    assert user.purge_codes() == 0
