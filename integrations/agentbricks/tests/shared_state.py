"""Create-once, share-everywhere resources for a pytest run, including all xdist workers.

Each resource is one JSON file in a directory every worker can reach, guarded by a per-resource
file lock (the pattern pytest-xdist documents). The first process to ask creates it while the others
wait; later processes read the cached value. Whatever a creator ``track``s lands in the same file,
which is how the controller finds what to destroy after every worker has finished.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pathlib
from collections.abc import Callable, Iterator
from typing import Any

from common import MatrixError

Track = Callable[..., None]


@contextlib.contextmanager
def _locked(path: pathlib.Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _read(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class SharedResources:
    def __init__(self, directory: pathlib.Path):
        self._directory = directory

    def get_or_create(self, key: str, create: Callable[[Track], dict[str, Any]]) -> dict[str, Any]:
        """The cached value for ``key``, calling ``create(track)`` in exactly one process.

        ``track(**fields)`` records what ``create`` has made so far, before it can fail, for
        ``tracked``. A failure is cached too, so other workers fail fast instead of retrying a
        deploy that takes minutes.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{key}.json"
        with _locked(self._directory / f"{key}.lock"):
            entry = _read(path)
            if "error" in entry:
                raise MatrixError(f"{key} setup already failed: {entry['error']}")
            if "value" in entry:
                return entry["value"]

            def track(**fields: Any) -> None:
                entry.setdefault("cleanup", {}).update(fields)
                path.write_text(json.dumps(entry), encoding="utf-8")

            try:
                entry["value"] = create(track)
            except Exception as exc:
                entry["error"] = str(exc)
                path.write_text(json.dumps(entry), encoding="utf-8")
                raise
            path.write_text(json.dumps(entry), encoding="utf-8")
            return entry["value"]

    def tracked(self) -> dict[str, dict[str, Any]]:
        """What each resource's creator recorded, for teardown."""
        entries = {path.stem: _read(path) for path in sorted(self._directory.glob("*.json"))}
        return {key: entry["cleanup"] for key, entry in entries.items() if entry.get("cleanup")}
