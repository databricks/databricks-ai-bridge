"""Native MLflow evaluation calls, sharing the tracing command's workspace adapter."""

from __future__ import annotations

import importlib
import io
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from numbers import Real
from typing import Any
from urllib.parse import quote

from databricks_agentbricks.clients.mlflow_trace_client import MLflowTraceClient


@dataclass(frozen=True)
class EvaluationFacts:
    metrics: dict[str, float]
    scored: int | None
    errors: int | None


class EvaluationClient(MLflowTraceClient):
    """Evaluate existing answers, never call the agent again."""

    def get_scorer(self, name: str, experiment_id: str) -> Any:
        importlib.import_module("pandas")  # Optional full-MLflow evaluation dependency.
        from mlflow.genai.scorers import get_scorer

        return get_scorer(name=name, experiment_id=experiment_id)

    @contextmanager
    def evaluation_run(self, experiment_id: str) -> Iterator[str]:
        with self.module.start_run(
            experiment_id=experiment_id, run_name="AgentBricks evaluation"
        ) as run:
            yield run.info.run_id

    def evaluate(self, data: list[dict[str, Any]], scorer: Any) -> EvaluationFacts:
        from mlflow.genai import evaluate

        # Native displays and scorer failures can include answer text. Keep CLI JSON and its
        # shareable evaluation logs content-free; full assessments remain in the MLflow run.
        previous_logging = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = evaluate(data=data, scorers=[scorer])
        finally:
            logging.disable(previous_logging)
        metrics = {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, Real) and math.isfinite(value)
        }
        frame = result.result_df
        if frame is None:
            return EvaluationFacts(metrics, None, None)
        values = [column for column in frame.columns if column.endswith("/value")]
        errors = [
            column for column in frame.columns if column.endswith(("/error_code", "/error_message"))
        ]
        scored_rows = frame[values].notna().any(axis=1) if values else None
        has_error = frame[errors].notna().any(axis=1) if errors else None
        if scored_rows is not None and has_error is not None:
            scored_rows &= ~has_error
        scored = int(scored_rows.sum()) if scored_rows is not None else 0
        failed = int(has_error.sum()) if has_error is not None else 0
        return EvaluationFacts(metrics, scored, failed)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.module.log_metrics(metrics)

    def browser_url(
        self, experiment_id: str, *, run_id: str | None = None, trace_id: str | None = None
    ) -> str | None:
        from databricks.sdk.core import Config

        try:
            host = Config(profile=self._profile).host
        except Exception:
            # A link is optional; auth re-resolution must not invalidate completed evaluations.
            return None
        if not host:
            return None
        base = f"{host.rstrip('/')}/ml/experiments/{quote(experiment_id, safe='')}"
        if run_id:
            return f"{base}/runs/{quote(run_id, safe='')}"
        if trace_id:
            return f"{base}/traces/{quote(trace_id.rsplit('/', 1)[-1], safe='')}"
        return base
