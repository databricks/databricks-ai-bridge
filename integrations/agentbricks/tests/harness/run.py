"""One run's identity: ``RunConfig``, the names every resource of the run carries, and constants.

``suites/conftest.py`` builds one ``RunConfig`` per pytest run; fixtures and tests read it through the
``run_config`` fixture. Resource names start with the run id, so leftover cleanup can find them.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import re
import tempfile

AUTHORING_PATHS = ("cli", "direct")
E2E_MODEL = "system.ai.gpt-5-2"
APP_PREFIX = "agent-bricks-"


class E2EError(RuntimeError):
    """A reproducible setup or execution failure."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def project_prefix(run_id: str) -> str:
    """Names every project of a run, so leftover cleanup can find the Apps a crashed test left."""
    return f"t-{run_id}-"


# App names are capped at 30 characters: "agent-bricks-t-" + run id + "-" + 6 random characters.
RUN_ID_PATTERN = re.compile(r"[a-z0-9]{1,8}")


def run_output_dir(run_id: str) -> pathlib.Path:
    """Where a run writes its logs, projects and ``apps.jsonl``; leftover cleanup finds it by the id alone."""
    return pathlib.Path(tempfile.gettempdir()) / f"agentbricks-e2e-{run_id}"


def scratch_schema_prefix(run_id: str) -> str:
    """Names every temporary schema of a run, so leftover cleanup can find ones a test left behind."""
    return f"ab_e2e_{run_id}_"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Values shared by every fixture of one pytest run, including all xdist workers."""

    run_id: str
    output: pathlib.Path
    # Operator override; None lets the workspace pick the first running warehouse.
    warehouse_id: str | None
    preprovisioned_app_catalog_access: bool
    wheel: pathlib.Path | None
