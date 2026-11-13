"""Bounded complete output retained for a failed subprocess."""

from __future__ import annotations

import os
import threading
import time
from typing import IO

# Leave room for the enclosing error message in the 4096-byte telemetry field.
_MAX_TAIL_BYTES = 3072
_DRAIN_GRACE_S = 0.25


class OutputTail:
    def __init__(self) -> None:
        self._data = b""
        self._overflowed = False

    def append(self, data: bytes) -> None:
        if self._overflowed:
            return
        if len(self._data) + len(data) > _MAX_TAIL_BYTES:
            # Partial captures can lose the context needed for downstream credential scrubbing.
            self._overflowed = True
            self._data = b""
            return
        self._data += data

    def text(self) -> str:
        if self._overflowed:
            return "[REDACTED]"
        return self._data.decode("utf-8", errors="replace")


class StreamRelay:
    """Forward a binary pipe to its original destination while retaining bounded diagnostics."""

    def __init__(self, stream: IO[bytes], destination: int) -> None:
        self.tail = OutputTail()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._forward_started: float | None = None
        self._forward_wait_seconds = 0.0
        self._drain_deadline: float | None = None
        # The reader owns its descriptor even after the subprocess owner closes the original.
        reader = os.fdopen(os.dup(stream.fileno()), "rb", buffering=0)
        try:
            if os.name == "posix":
                os.set_blocking(reader.fileno(), False)
            self._thread = threading.Thread(
                target=self._copy, args=(reader, destination), daemon=True
            )
            self._thread.start()
        except Exception:
            reader.close()
            raise

    def _copy(self, stream: IO[bytes], destination: int) -> None:
        forward = True
        try:
            while not self._stop.is_set():
                with self._state_lock:
                    if (
                        self._drain_deadline is not None
                        and time.monotonic() - self._forward_wait_seconds >= self._drain_deadline
                    ):
                        return
                try:
                    data = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    self._stop.wait(0.01)
                    continue
                if not data:
                    return
                with self._state_lock:
                    if self._stop.is_set():
                        return
                    self.tail.append(data)
                    started = self._forward_started = time.monotonic()
                    cpu_started = time.thread_time()
                try:
                    view = memoryview(data)
                    while view and forward and not self._stop.is_set():
                        try:
                            view = view[os.write(destination, view) :]
                        except OSError:
                            # Keep draining so a closed destination cannot block the child.
                            forward = False
                finally:
                    # Charge write CPU work while exempting waits on the destination or scheduler.
                    waiting = max(
                        0.0, time.monotonic() - started - (time.thread_time() - cpu_started)
                    )
                    with self._state_lock:
                        self._forward_wait_seconds += waiting
                        self._forward_started = None
        except (OSError, ValueError):
            return
        finally:
            stream.close()

    def cancel(self) -> None:
        """Abandon pending output without waiting for a blocked destination."""
        self._stop.set()

    def finish(self) -> None:
        # Bound processing while allowing blocked output forwarding to finish.
        with self._state_lock:
            now = time.monotonic()
            started = self._forward_started if self._forward_started is not None else now
            deadline = self._drain_deadline = started - self._forward_wait_seconds + _DRAIN_GRACE_S
        while self._thread.is_alive() and not self._stop.is_set():
            self._thread.join(timeout=0.05)
            with self._state_lock:
                if (
                    self._forward_started is None
                    and time.monotonic() - self._forward_wait_seconds >= deadline
                ):
                    self._stop.set()
                    return
