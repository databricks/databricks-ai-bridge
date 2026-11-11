"""Bounded in-memory output retained for a failed subprocess."""

from __future__ import annotations

import os
import re
import threading
import time
from typing import IO

# Leave room for the enclosing error message in the 4096-byte telemetry field.
_MAX_TAIL_BYTES = 3072
_PEM_PREFIX = b"-----BEGIN "
_REDACTION_CONTEXT = re.compile(rb"""(?i)[:={"'\[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|bearer|basic""")
_DRAIN_GRACE_S = 0.25


class OutputTail:
    def __init__(self) -> None:
        self._data = b""
        self._truncated = False
        self._marker_suffix = b""
        self._pem_seen = False
        self._context_seen = False

    def append(self, data: bytes) -> None:
        if self._pem_seen:
            return
        scanned = self._marker_suffix + data
        if _PEM_PREFIX in scanned:
            # Omit PEM-bearing diagnostics even when retention would discard their opening marker.
            self._pem_seen = True
            self._data = b""
            return
        self._marker_suffix = scanned[-(len(_PEM_PREFIX) - 1) :]
        self._context_seen |= _REDACTION_CONTEXT.search(scanned) is not None
        self._truncated |= len(self._data) + len(data) > _MAX_TAIL_BYTES
        self._data = (self._data + data)[-_MAX_TAIL_BYTES:]

    def text(self) -> str:
        # Credential values can outlive their identifying key or auth scheme when bytes are lost.
        if self._pem_seen or (self._truncated and self._context_seen):
            return "[REDACTED]"
        data = self._data
        if self._truncated:
            # Discard a partial leading line so credential names stay with their values.
            _, _, data = data.partition(b"\n")
        return data.decode("utf-8", errors="replace")


class StreamRelay:
    """Forward a binary pipe to its original destination while retaining a bounded tail."""

    def __init__(self, stream: IO[bytes], destination: int) -> None:
        self.tail = OutputTail()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._forward_started: float | None = None
        self._forward_seconds = 0.0
        self._drain_deadline: float | None = None
        self._thread = threading.Thread(target=self._copy, args=(stream, destination), daemon=True)
        self._thread.start()

    def _copy(self, stream: IO[bytes], destination: int) -> None:
        forward = True
        try:
            while not self._stop.is_set():
                with self._state_lock:
                    if (
                        self._drain_deadline is not None
                        and time.monotonic() - self._forward_seconds >= self._drain_deadline
                    ):
                        return
                data = os.read(stream.fileno(), 65536)
                if not data:
                    return
                with self._state_lock:
                    if self._stop.is_set():
                        return
                    started = self._forward_started = time.monotonic()
                try:
                    self.tail.append(data)
                    view = memoryview(data)
                    while view and forward:
                        try:
                            view = view[os.write(destination, view) :]
                        except OSError:
                            # Keep draining so a closed destination cannot block the child.
                            forward = False
                finally:
                    with self._state_lock:
                        self._forward_seconds += time.monotonic() - started
                        self._forward_started = None
        except (OSError, ValueError):
            return

    def finish(self) -> None:
        # Bound descendant reads without charging time spent forwarding to slow consumers.
        with self._state_lock:
            now = time.monotonic()
            started = self._forward_started if self._forward_started is not None else now
            deadline = self._drain_deadline = started - self._forward_seconds + _DRAIN_GRACE_S
        while self._thread.is_alive():
            self._thread.join(timeout=0.05)
            with self._state_lock:
                if (
                    self._forward_started is None
                    and time.monotonic() - self._forward_seconds >= deadline
                ):
                    self._stop.set()
                    return
