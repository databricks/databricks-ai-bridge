"""Local TCP port checks and picks for the servers ``agentbricks dev`` starts."""

from __future__ import annotations

import errno
import socket
import sys

# The generated runtime serves with uvicorn on the IPv4 wildcard address (see the templates'
# ``runtime/main.py``), so that is the bind this check reproduces.
_APP_BIND_HOST = "0.0.0.0"
_LOOPBACK_HOSTS = ("127.0.0.1", "::1")
_CONNECT_TIMEOUT_S = 0.5


def port_in_use(port: int) -> bool:
    """Whether a local server already holds ``port``, so the app could not listen on it.

    Two probes, so neither socket-option semantics nor address family causes a wrong answer:

    - Bind the address the app binds, with the same ``SO_REUSEADDR`` uvicorn sets. On POSIX that
      option still refuses a port another socket is listening on, but accepts one left in
      ``TIME_WAIT`` by a server that just exited, which the app could reuse too. Windows is
      different: there the option would let this bind succeed over a live listener, so it is skipped.
    - Connect to IPv4 and IPv6 loopback, to catch a server bound only to localhost, which a wildcard
      bind with ``SO_REUSEADDR`` can coexist with on macOS but which would still answer the user's
      ``localhost`` requests.

    Only "address in use" counts as busy. Any other bind error (for example a privileged port) is
    left for the app to report.
    """
    return not _can_bind(port) or any(_accepts_connections(host, port) for host in _LOOPBACK_HOSTS)


def free_port() -> int:
    """A port the OS reports free on the app's bind address, for the app behind run-local's proxy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_APP_BIND_HOST, 0))
        return sock.getsockname()[1]


def _can_bind(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((_APP_BIND_HOST, port))
        except OSError as exc:
            return exc.errno != errno.EADDRINUSE
    return True


def _accepts_connections(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_CONNECT_TIMEOUT_S):
            return True
    except OSError:
        # Refused, unreachable, or IPv6 unavailable: nothing is answering on this address.
        return False
