"""Minimal dotenv reader for the CLI and examples (no python-dotenv dependency)."""

from __future__ import annotations

import os


def read_env_file(path: str | None) -> dict[str, str]:
    """``KEY=value`` lines from ``path`` (``export`` prefixes, quotes and `` #`` comments stripped)."""
    if not path:
        return {}
    values: dict[str, str] = {}
    with open(os.path.expanduser(path)) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ").strip()] = (
                value.split(" #", 1)[0].strip().strip("'\"")
            )
    return values
