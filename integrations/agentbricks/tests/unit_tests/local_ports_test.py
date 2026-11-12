"""Real-socket coverage for the ``agentbricks dev`` port check."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from databricks_agentbricks.clients.local_ports import free_port, port_in_use, route_exists


@contextmanager
def _listener(family: socket.AddressFamily, host: str) -> Iterator[int]:
    """Listen on an ephemeral port of ``host`` and yield the port."""
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, 0))
        sock.listen()
        yield sock.getsockname()[1]


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


@pytest.mark.parametrize("host", ["0.0.0.0", "127.0.0.1"])
def test_ipv4_listener_is_in_use(host: str) -> None:
    with _listener(socket.AF_INET, host) as port:
        assert port_in_use(port)


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="IPv6 loopback unavailable")
def test_ipv6_only_loopback_listener_is_in_use() -> None:
    with _listener(socket.AF_INET6, "::1") as port:
        assert port_in_use(port)


def test_closed_port_is_free() -> None:
    with _listener(socket.AF_INET, "127.0.0.1") as port:
        pass
    assert not port_in_use(port)


def test_free_port_returns_a_bindable_port():
    port = free_port()
    assert 0 < port < 65536
    assert not port_in_use(port)


@contextmanager
def _serving(app) -> Iterator[int]:
    """Serve ``app`` with uvicorn on a free local port for the duration of the block."""
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_route_exists_distinguishes_a_post_only_route_from_a_missing_one():
    calls: list[str] = []

    async def invoke(request):
        calls.append(request.method)
        return JSONResponse({})

    app = Starlette(routes=[Route("/api/invocations", invoke, methods=["POST"])])
    with _serving(app) as port:
        assert route_exists(port, "/api/invocations") is True  # 405: served, POST only
        assert route_exists(port, "/invocations") is False  # 404
    assert calls == []  # the probe never reaches the handler


def test_route_exists_is_none_when_nothing_answers():
    with _listener(socket.AF_INET, "127.0.0.1") as port:
        pass
    assert route_exists(port, "/api/invocations") is None


@pytest.mark.parametrize("status", [401, 403, 302, 500, 503])
def test_route_exists_is_inconclusive_for_auth_redirect_and_server_errors(status: int):
    async def answer(request):
        return Response(status_code=status, headers={"location": "/elsewhere"})

    app = Starlette(routes=[Route("/api/invocations", answer, methods=["GET"])])
    with _serving(app) as port:
        assert route_exists(port, "/api/invocations") is None


class _AuthGate:
    """Rejects unauthenticated requests before any routing, as an app's auth middleware does."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and b"authorization" not in dict(scope["headers"]):
            await Response("unauthorized", status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def test_route_exists_does_not_mistake_an_auth_gate_for_a_route():
    async def invoke(request):
        return JSONResponse({})

    app = _AuthGate(Starlette(routes=[Route("/invocations", invoke, methods=["POST"])]))
    with _serving(app) as port:
        assert route_exists(port, "/api/invocations") is None
        assert route_exists(port, "/invocations") is None
