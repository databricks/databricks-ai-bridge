"""Matrix-agnostic pieces shared by the test modules and ``run_suites.py``.

``run_suites.py`` provisions the workspace dependencies and passes their resolved values in a JSON
file named by ``AGENTBRICKS_TEST_INPUTS``. This module loads that file, owns the matrix constants
(so the runner and the tests cannot drift), and records the evidence rows the runner turns into
``evidence.json``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import json
import os
import pathlib
import shlex
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

INPUTS_ENV = "AGENTBRICKS_TEST_INPUTS"

FRAMEWORKS = ("langgraph",)
AUTHORING_PATHS = ("cli", "direct")
RUNTIMES = ("dev", "deploy")
TOOL_KINDS = ("sandbox", "mcp", "python", "uc_function", "genie")
E2E_MODEL = "system.ai.gpt-5-2"
TOOL_RESOURCE_PREFIX = "agentbricks-tool-"

# The (authoring, runtime, kind) evidence cells each matrix must produce a passing row for. The
# runner's verify step reads this, keyed by suite name.
MATRIX_CELLS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "tools": tuple(
        (authoring, runtime, kind)
        for authoring in AUTHORING_PATHS
        for runtime in RUNTIMES
        for kind in TOOL_KINDS
    )
    + tuple((authoring, "deploy", "selection") for authoring in AUTHORING_PATHS),
    "session_store": (("cli", "deploy", "session_store"),),
    "memory_store": (("cli", "deploy", "memory_store"),),
    "tracing": (("cli", "deploy", "tracing"),),
}

EXPECTED = {
    "sandbox": "the exact hidden marker read from the temporary UC volume file",
    "python": "AGENTBRICKS_PYTHON_OK",
    "uc_function": "AGENTBRICKS_UC_OK:matrix",
    "mcp": "a web-search tool call and a non-empty https result",
    "genie": "a genie_ask tool call and a non-empty Genie response",
    "selection": "the UC function tool's exact result, and no call to the Python marker tool",
    "session_store": "the declared session store exists",
    "memory_store": "the declared memory store exists",
    "tracing": "the tracing experiment is bound",
}


class MatrixError(RuntimeError):
    """A reproducible setup or execution failure."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def last_lines(path: pathlib.Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def last_nonempty_line(path: pathlib.Path) -> str:
    for line in reversed(last_lines(path, 20).splitlines()):
        if line.strip():
            return line.strip()[:300]
    return "no output yet"


class Transcript:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        line = text.rstrip() + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def command(self, argv: Sequence[str], cwd: pathlib.Path | None = None) -> None:
        prefix = f"cd {shlex.quote(str(cwd))} && " if cwd else ""
        self.write(f"$ {prefix}{shlex.join(list(argv))}")

    def file_step(self, path: pathlib.Path, description: str) -> None:
        self.write(f"# write {path}: {description}")


@dataclasses.dataclass(frozen=True)
class Inputs:
    """The workspace dependencies ``run_suites.py`` resolved for this suite run."""

    suite: str
    databricks_profile: str | None
    app_auth_profile: str | None
    preprovisioned_app_catalog_access: bool
    bridge_sha: str | None
    wheel: pathlib.Path
    agentbricks_bin: pathlib.Path
    output: pathlib.Path
    run_suffix: str
    catalog: str
    scratch_schema: str
    uc_function: str
    transitive_uc_function: str
    uc_volume: str
    volume_marker: str
    volume_file_path: str
    genie_space_id: str
    warehouse_id: str

    @classmethod
    def from_env(cls) -> Inputs:
        path = os.environ.get(INPUTS_ENV)
        if not path:
            raise MatrixError(f"{INPUTS_ENV} is not set; run these tests through run_suites.py.")
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            suite=data["suite"],
            databricks_profile=data.get("databricks_profile"),
            app_auth_profile=data.get("app_auth_profile") or data.get("databricks_profile"),
            preprovisioned_app_catalog_access=bool(data.get("preprovisioned_app_catalog_access")),
            bridge_sha=data.get("bridge_sha"),
            wheel=pathlib.Path(data["wheel"]),
            agentbricks_bin=pathlib.Path(data["agentbricks_bin"]),
            output=pathlib.Path(data["output_dir"]),
            run_suffix=data["run_suffix"],
            catalog=data["catalog"],
            scratch_schema=data["schema"],
            uc_function=data["uc_function"],
            transitive_uc_function=data["transitive_uc_function"],
            uc_volume=data["uc_volume"],
            volume_marker=data["volume_marker"],
            volume_file_path=data["volume_file_path"],
            genie_space_id=data["genie_space_id"],
            warehouse_id=data["warehouse_id"],
        )


@dataclasses.dataclass
class EvidenceRow:
    framework: str
    authoring: str
    runtime: str
    tool_kind: str
    status: str
    command: str
    expected: str
    actual: str
    duration_seconds: float
    artifact_paths: list[str]
    app_name: str | None = None
    app_url: str | None = None
    error: str | None = None


class Evidence:
    """Appends the files under the output directory that the runner turns into ``evidence.json``."""

    def __init__(self, output: pathlib.Path):
        self.rows_path = output / "rows.jsonl"
        self.registry_path = output / "registry.jsonl"
        self.cleanup_path = output / "cleanup.jsonl"
        self.grant_dir = output / "grant-checks"
        self._lock = threading.Lock()

    def register_app(self, name: str) -> None:
        self._append(self.registry_path, {"kind": "app", "app_name": name})

    def register_project(self, app_name: str, **fields: Any) -> None:
        self._append(self.registry_path, {"kind": "project", "app_name": app_name, **fields})

    def mark_cleaned(self, app_name: str, results: Sequence[dict[str, Any]]) -> None:
        for result in results:
            self._append(self.cleanup_path, result)
        self._append(self.registry_path, {"kind": "cleaned", "app_name": app_name})

    def record_row(self, row: EvidenceRow) -> None:
        self._append(self.rows_path, dataclasses.asdict(row))

    def read_rows(self) -> list[dict[str, Any]]:
        if not self.rows_path.exists():
            return []
        with self._lock:
            lines = self.rows_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def record_setup_failure(
        self,
        authoring: str,
        runtime: str,
        exc: Exception,
        log_path: pathlib.Path | None,
        kinds: Sequence[str],
        app_name: str | None = None,
    ) -> None:
        """Fail the cells whose agent could not be set up, so the evidence names them."""
        existing = {
            row["tool_kind"]
            for row in self.read_rows()
            if row["authoring"] == authoring and row["runtime"] == runtime
        }
        for kind in kinds:
            if kind in existing:
                continue
            self.record_row(
                EvidenceRow(
                    framework=FRAMEWORKS[0],
                    authoring=authoring,
                    runtime=runtime,
                    tool_kind=kind,
                    status="fail",
                    command="runtime setup",
                    expected=EXPECTED[kind],
                    actual="",
                    duration_seconds=0.0,
                    artifact_paths=[str(log_path)] if log_path else [],
                    app_name=app_name,
                    error=str(exc),
                )
            )

    def attach_artifact_to_failed_rows(self, app_name: str, artifact: pathlib.Path) -> bool:
        rows = self.read_rows()
        failed = [
            row for row in rows if row["status"] == "fail" and row.get("app_name") == app_name
        ]
        for row in failed:
            row["artifact_paths"].append(str(artifact))
        if failed:
            with self._lock:
                temporary = self.rows_path.with_suffix(".jsonl.tmp")
                temporary.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                os.replace(temporary, self.rows_path)
        return bool(failed)

    def record_grant_check(self, app_name: str, **fields: Any) -> None:
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        path = self.grant_dir / f"{app_name}.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update(fields)
        current["app_name"] = app_name
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def _append(self, path: pathlib.Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record) + "\n")


@dataclasses.dataclass
class Outcome:
    """What a cell observed; the row's ``actual`` and links are derived from it."""

    serialized: str = ""
    command: str = ""
    log_path: pathlib.Path | None = None
    app_name: str | None = None
    app_url: str | None = None


@contextlib.contextmanager
def recorded_row(
    evidence: Evidence,
    authoring: str,
    runtime: str,
    kind: str,
    *,
    command: str,
    expected: str,
    marker: str | None = None,
) -> Iterator[Outcome]:
    """Run one matrix cell and append its pass/fail evidence row whatever happens."""
    outcome = Outcome(command=command)
    started = time.monotonic()
    error: str | None = None
    try:
        yield outcome
    except Exception as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        evidence.record_row(
            EvidenceRow(
                framework=FRAMEWORKS[0],
                authoring=authoring,
                runtime=runtime,
                tool_kind=kind,
                status="pass" if error is None else "fail",
                command=outcome.command,
                expected=expected,
                actual=evidence_excerpt(outcome.serialized, marker),
                duration_seconds=round(time.monotonic() - started, 3),
                artifact_paths=[str(outcome.log_path)] if outcome.log_path else [],
                app_name=outcome.app_name,
                app_url=outcome.app_url,
                error=error,
            )
        )


def evidence_excerpt(serialized: str, required_marker: str | None) -> str:
    limit = 6000
    if len(serialized) <= limit:
        return serialized
    head_length = limit // 2
    head = serialized[:head_length]
    if required_marker is None or required_marker in head:
        tail = serialized[-head_length:]
    else:
        marker_offset = serialized.find(required_marker)
        start = max(head_length, marker_offset - head_length // 2)
        tail = serialized[start : start + head_length]
    return f"{head}\n... response truncated ...\n{tail}"
