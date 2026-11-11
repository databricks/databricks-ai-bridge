"""Bounded failure facts shared by CLI workflows and their telemetry adapter."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class FailureReason(str, Enum):
    """Internal classification hints; these are not a wire schema or an exhaustive taxonomy."""

    PROJECT_INVALID = "PROJECT_INVALID"
    PORT_IN_USE = "PORT_IN_USE"
    UV_MISSING = "UV_MISSING"
    PYTHON_UNAVAILABLE = "PYTHON_UNAVAILABLE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    SUBPROCESS_FAILED = "SUBPROCESS_FAILED"
    CHILD_EXIT_BEFORE_LISTENING = "CHILD_EXIT_BEFORE_LISTENING"
    CHILD_EXIT_AFTER_LISTENING = "CHILD_EXIT_AFTER_LISTENING"
    APP_READINESS_TIMEOUT = "APP_READINESS_TIMEOUT"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    FEATURE_DISABLED = "FEATURE_DISABLED"


class FailureStage(str, Enum):
    PROJECT = "PROJECT"
    TRACING = "TRACING"
    RUN_LOCAL = "RUN_LOCAL"
    LOCAL_AGENT = "LOCAL_AGENT"
    CLEANUP = "CLEANUP"
    PREFLIGHT = "PREFLIGHT"
    APP_START = "APP_START"
    APP_READY = "APP_READY"
    RECONCILE = "RECONCILE"
    RUNTIME_STORE = "RUNTIME_STORE"
    APP_YAML = "APP_YAML"
    SYNC = "SYNC"
    COMPUTE_WAIT = "COMPUTE_WAIT"
    APP_UPDATE = "APP_UPDATE"
    ROLLOUT = "ROLLOUT"
    GRANTS = "GRANTS"
    READY = "READY"


class Readiness(str, Enum):
    ANSWERED = "ANSWERED"
    BOOTING = "BOOTING"
    UNVERIFIABLE = "UNVERIFIABLE"
    UNREACHABLE = "UNREACHABLE"


@dataclass
class CommandDiagnostics:
    failure_stage: FailureStage | None = None
    app_listened: bool | None = None
    time_to_listen_ms: int | None = None
    prepare_environment: bool | None = None
    subprocess_exit_code: int | None = None
    readiness: Readiness | None = None
    probe_http_status: int | None = None
    _run_started: float | None = None

    def starting_local_agent(self) -> None:
        self.app_listened = False
        self._run_started = time.monotonic()

    def local_agent_listened(self) -> None:
        self.app_listened = True
        if self._run_started is not None:
            self.time_to_listen_ms = max(0, int((time.monotonic() - self._run_started) * 1000))

    @contextmanager
    def phase(self, stage: FailureStage) -> Iterator[None]:
        try:
            yield
        except BaseException:
            # An outer workflow or cleanup must not replace the more specific failed phase.
            if self.failure_stage is None:
                self.failure_stage = stage
            raise
