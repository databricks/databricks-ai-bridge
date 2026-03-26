"""Settings for LongRunningAgentServer."""

from dataclasses import dataclass


@dataclass
class LongRunningSettings:
    """Configuration for :class:`LongRunningAgentServer`.

    All values have sensible defaults. Callers override individual fields at
    construction time — environment-variable reading is the caller's concern.
    """

    task_timeout_seconds: float = 3600.0
    poll_interval_seconds: float = 1.0
    db_statement_timeout_ms: int = 5000
    cleanup_timeout_seconds: float = 7.0

    def __post_init__(self) -> None:
        if self.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.db_statement_timeout_ms <= 0:
            raise ValueError("db_statement_timeout_ms must be positive")
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be positive")
        db_timeout_s = self.db_statement_timeout_ms / 1000.0
        if self.cleanup_timeout_seconds <= db_timeout_s:
            raise ValueError(
                f"cleanup_timeout_seconds ({self.cleanup_timeout_seconds}) must be "
                f"strictly greater than db_statement_timeout_ms converted to seconds "
                f"({db_timeout_s})"
            )
