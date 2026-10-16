"""Real-socket coverage for the ``agentbricks dev`` port check."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from databricks_agentbricks.clients.local_ports import port_in_use


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
