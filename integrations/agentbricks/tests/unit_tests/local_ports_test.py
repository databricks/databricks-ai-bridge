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
from starlette.responses import JSONResponse
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
