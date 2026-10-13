"""Test-log output: every command the suites run is echoed, so a run's log replays it."""

from __future__ import annotations

import pathlib
import shlex
import sys
from collections.abc import Sequence


def log(text: str) -> None:
    sys.stdout.write(text.rstrip() + "\n")
    sys.stdout.flush()


def log_command(argv: Sequence[str], cwd: pathlib.Path | None = None) -> None:
    prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
    log(f"$ {prefix}{shlex.join(list(argv))}")


def last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"
