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
    heartbeat_interval_seconds: float = 3.0
    heartbeat_stale_threshold_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.db_statement_timeout_ms <= 0:
            raise ValueError("db_statement_timeout_ms must be positive")
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.heartbeat_stale_threshold_seconds <= 0:
            raise ValueError("heartbeat_stale_threshold_seconds must be positive")
        if self.heartbeat_stale_threshold_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                f"heartbeat_stale_threshold_seconds ({self.heartbeat_stale_threshold_seconds}) "
                f"must be strictly greater than heartbeat_interval_seconds "
                f"({self.heartbeat_interval_seconds}) to avoid false stale-run detection."
            )
        db_timeout_s = self.db_statement_timeout_ms / 1000.0
        if self.cleanup_timeout_seconds <= db_timeout_s:
            raise ValueError(
                f"cleanup_timeout_seconds ({self.cleanup_timeout_seconds}) must be "
                f"strictly greater than db_statement_timeout_ms converted to seconds "
                f"({db_timeout_s})"
            )
