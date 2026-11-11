"""Bounded in-memory output retained for a failed subprocess."""

from __future__ import annotations

import os
import threading
from typing import IO

# Leave room for the enclosing error message in the 4096-byte telemetry field.
_MAX_TAIL_BYTES = 3072


class OutputTail:
    def __init__(self) -> None:
        self._data = b""
        self._truncated = False

    def append(self, data: bytes) -> None:
        self._truncated |= len(self._data) + len(data) > _MAX_TAIL_BYTES
        self._data = (self._data + data)[-_MAX_TAIL_BYTES:]

    def text(self) -> str:
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
        self._thread = threading.Thread(target=self._copy, args=(stream, destination), daemon=True)
        self._thread.start()

    def _copy(self, stream: IO[bytes], destination: int) -> None:
        forward = True
        try:
            while not self._stop.is_set():
                data = os.read(stream.fileno(), 65536)
                if not data or self._stop.is_set():
                    return
                self.tail.append(data)
                view = memoryview(data)
                while view and forward:
                    try:
                        view = view[os.write(destination, view) :]
                    except OSError:
                        # Keep draining so a closed output destination cannot block the child.
                        forward = False
        except (OSError, ValueError):
            return

    def finish(self) -> None:
        # Descendants may inherit a pipe after the direct child exits.
        self._thread.join(timeout=0.25)
        self._stop.set()
