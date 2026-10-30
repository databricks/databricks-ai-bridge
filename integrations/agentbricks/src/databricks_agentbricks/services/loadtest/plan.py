"""Normalized, immutable inputs for an endpoint load test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from databricks_agentbricks.errors import AgentCliError

if TYPE_CHECKING:
    from databricks_agentbricks.services.endpoint_service import EndpointTarget


def load_bodies(requests_file: Path | None) -> tuple[str, ...] | None:
    """Read and normalize the optional array exactly once, before workspace access."""
    if requests_file is None:
        return None
    try:
        values = json.loads(requests_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentCliError(
            "--requests must contain a valid JSON array of request bodies."
        ) from exc
    if not isinstance(values, list) or not values:
        raise AgentCliError("--requests must contain a nonempty JSON array of request bodies.")
    return tuple(json.dumps(body) for body in values)


@dataclass(frozen=True)
class LoadTestPlan:
    """Keep one body source: serialized JSON templates, or ``None`` for no body."""

    target: EndpointTarget
    bodies: tuple[str, ...] | None = None
    concurrency: int = 1
    duration: int = 60
    max_requests: int | None = None
    report_dir: Path | None = None
    responses_file: Path | None = None
    warmup: bool = True

    def validate(self) -> None:
        """Validate local limits and output paths before resolving a workspace target."""
        if (
            self.concurrency < 1
            or self.duration < 1
            or (self.max_requests is not None and self.max_requests < 1)
        ):
            raise AgentCliError(
                "Load test concurrency, duration, and request limit must be positive."
            )
        if self.bodies is not None and not self.bodies:
            raise AgentCliError("--requests must contain a nonempty JSON array of request bodies.")
        if self.responses_file is not None:
            responses = self.responses_file
            if responses.exists() or responses.is_symlink():
                raise AgentCliError(
                    "--save-responses requires a new file; existing files are preserved."
                )
            if not responses.parent.is_dir():
                raise AgentCliError("The parent directory of --save-responses must exist.")
            if self.report_dir is not None and responses.resolve().is_relative_to(
                self.report_dir.resolve()
            ):
                raise AgentCliError(
                    "--save-responses must be outside --report-dir; it contains request and answer content."
                )
        if self.report_dir is not None:
            report = self.report_dir
            try:
                if report.exists() and (not report.is_dir() or any(report.iterdir())):
                    raise AgentCliError("--report-dir must be a new or empty directory.")
                report.mkdir(parents=True, exist_ok=True, mode=0o700)
            except OSError as exc:
                raise AgentCliError("Could not prepare --report-dir.") from exc
