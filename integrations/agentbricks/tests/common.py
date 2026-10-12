"""Constants, ``RunConfig`` and logging/subprocess helpers shared by the E2E suites.

``run_context.py`` builds one ``RunConfig`` per pytest run; fixtures and tests read it through the
``run_config`` fixture.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence

FRAMEWORKS = ("langgraph",)
AUTHORING_PATHS = ("cli", "direct")
E2E_MODEL = "system.ai.gpt-5-2"
TOOL_RESOURCE_PREFIX = "agentbricks-tool-"
APP_PREFIX = "agent-bricks-"


class E2EError(RuntimeError):
    """A reproducible setup or execution failure."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def project_prefix(run_id: str) -> str:
    """Names every project of a run, so teardown can sweep the Apps a crashed test leaked."""
    return f"t-{run_id}-"


# App names are capped at 30 characters: "agent-bricks-t-" + run id + "-" + 6 random characters.
RUN_ID_PATTERN = re.compile(r"[a-z0-9]{1,8}")


def run_output_dir(run_id: str) -> pathlib.Path:
    """Where a run writes its logs, projects and ``apps.jsonl``; the sweep finds it by the id alone."""
    return pathlib.Path(tempfile.gettempdir()) / f"agentbricks-e2e-{run_id}"


def scratch_schema_prefix(run_id: str) -> str:
    """Names every temporary schema of a run, so the controller can sweep ones a test left behind."""
    return f"ab_e2e_{run_id}_"


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


def run_command(
    argv: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 300,
) -> str:
    """Run a command, echo its output, and return stdout; a non-zero exit is a ``E2EError``."""
    log_command(argv, cwd)
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    for text in (result.stdout, result.stderr):
        if text.strip():
            log(text)
    if result.returncode != 0:
        raise E2EError(
            f"Command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"{result.stderr or result.stdout}"
        )
    return result.stdout


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Values shared by every fixture of one pytest run, including all xdist workers."""

    run_id: str
    output: pathlib.Path
    app_auth_profile: str | None
    # Operator override; None lets the workspace pick the first running warehouse.
    warehouse_id: str | None
    preprovisioned_app_catalog_access: bool
    bridge_sha: str | None
    wheel: pathlib.Path | None
