"""Evaluate saved agent answers and profile their explicitly correlated MLflow traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from databricks_agentbricks.clients.evaluation_client import EvaluationClient
from databricks_agentbricks.errors import AgentCliError

_SPAN_TYPES = {
    "LLM",
    "CHAT_MODEL",
    "CHAIN",
    "AGENT",
    "TOOL",
    "RETRIEVER",
    "EMBEDDING",
    "RERANKER",
    "PARSER",
    "UNKNOWN",
    "WORKFLOW",
    "TASK",
    "GUARDRAIL",
    "EVALUATOR",
}


_MAX_DATA_BYTES = 64 * 1024 * 1024


def _invalid_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def read_evaluation_json(path: Path) -> Any:
    """Read bounded private input without reflecting its contents in errors."""
    try:
        with path.open("rb") as source:
            content = source.read(_MAX_DATA_BYTES + 1)
        if len(content) > _MAX_DATA_BYTES:
            raise ValueError("input exceeds size limit")
        return json.loads(content.decode("utf-8"), parse_constant=_invalid_json_constant)
    except (OSError, ValueError, RecursionError) as exc:
        raise AgentCliError("Cannot read evaluation input as JSON (maximum 64 MiB).") from exc


def load_evaluation_data(path: Path, *, max_samples: int = 20) -> tuple[int, list[dict[str, Any]]]:
    """Validate a saved-response artifact before resolving any workspace resources."""
    data = read_evaluation_json(path)
    valid = (
        isinstance(data, dict)
        and type(data.get("schema_version")) is int
        and data["schema_version"] == 1
    )
    case_count = data.get("case_count") if isinstance(data, dict) else None
    observations = data.get("observations") if isinstance(data, dict) else None
    if (
        not valid
        or type(case_count) is not int
        or case_count < 1
        or not isinstance(observations, list)
        or max_samples < 1
    ):
        raise AgentCliError(
            "Invalid evaluation data: expected a version 1 saved-response artifact."
        )
    for observation in observations:
        if (
            not isinstance(observation, dict)
            or type(observation.get("case_index")) is not int
            or not 0 <= observation["case_index"] < case_count
            or "inputs" not in observation
            or "outputs" not in observation
            or not (observation.get("trace_id") is None or isinstance(observation["trace_id"], str))
        ):
            raise AgentCliError(
                "Invalid evaluation observation: require inputs, outputs, a valid case_index, "
                "and an optional string trace_id."
            )
    return case_count, observations[:max_samples]


class EvaluationService:
    def __init__(
        self,
        profile: str | None,
        experiment_id: str,
        *,
        scorer_names: tuple[str, ...] = (),
        traces: bool = False,
        max_samples: int = 20,
        expectations: list[dict[str, Any]] | None = None,
        warehouse_id: str | None = None,
        client: EvaluationClient | None = None,
    ):
        self.client = client or EvaluationClient(profile)
        self.experiment_id = experiment_id
        self.scorer_names = tuple(dict.fromkeys(scorer_names))
        self.traces = traces
        self.max_samples = max_samples
        self.expectations = expectations
        self.warehouse_id = warehouse_id
        self._scorers: list[Any] | None = None

    def _url(self, **identifier: str) -> str | None:
        try:
            return self.client.browser_url(self.experiment_id, **identifier)
        except Exception:
            return None  # Optional links must not invalidate successful reads or scores.

    def prepare(self) -> None:
        """Validate dependencies, experiment, warehouse, and scorers before analysis."""
        if self._scorers is not None:
            return
        if not self.experiment_id or self.max_samples < 1:
            raise AgentCliError("MLflow requires an experiment id and a positive sample limit.")
        if self.expectations is not None and (
            not isinstance(self.expectations, list)
            or not all(isinstance(item, dict) for item in self.expectations)
        ):
            raise AgentCliError(
                "MLflow expectations must be a JSON array of objects, one per case."
            )
        try:
            self.client.select_workspace()
            experiment = self.client.experiment_by_id(self.experiment_id)
            if experiment is None:
                raise AgentCliError("The selected MLflow experiment does not exist.")
            if self.traces:
                warehouse = self.client.resolve_warehouse_id(self.warehouse_id)
                self.client.require_warehouse_for_uc_read(experiment, warehouse)
            self._scorers = [
                self.client.get_scorer(name, self.experiment_id) for name in self.scorer_names
            ]
        except AgentCliError:
            raise
        except ImportError as exc:
            raise AgentCliError(
                "MLflow evaluation dependencies are required.",
                hint="Install 'databricks-agentbricks[evaluation]'.",
            ) from exc
        except Exception as exc:
            raise AgentCliError(
                f"MLflow preflight failed ({type(exc).__name__}).",
                hint="Check the selected profile, experiment permissions, and registered scorers.",
            ) from exc

    def finish(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        selected = observations[: self.max_samples]
        traces: dict[str, Any] = {
            "status": "disabled",
            "selected": len(selected),
            "correlated": 0,
            "fetched": 0,
            "missing": 0,
            "errors": 0,
            "coverage": None,
            "spans": [],
            "urls": [],
            "error_types": [],
        }
        evaluation: dict[str, Any] = {
            "status": "disabled",
            "run_id": None,
            "run_url": None,
            "scorers": [],
        }
        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": self.experiment_id,
            "sample_limit": self.max_samples,
            "sampled_successes": len(selected),
            "evaluation": evaluation,
            "traces": traces,
        }
        try:
            self.prepare()
            self._analyze(selected, traces, evaluation)
        except KeyboardInterrupt:
            summary["status"] = "cancelled"
            if self.traces:
                traces.update(
                    status="cancelled",
                    missing=len(selected) - traces["fetched"],
                    coverage=traces["fetched"] / len(selected) if selected else None,
                )
            if self.scorer_names:
                evaluation["status"] = "cancelled"
                for outcome in evaluation["scorers"]:
                    if outcome["status"] == "running":
                        outcome["status"] = "cancelled"
        except AgentCliError:
            raise
        if summary["status"] != "cancelled" and (
            evaluation["status"] in {"failed", "partial"} or traces["errors"]
        ):
            summary["status"] = "failed"
        return summary

    def _analyze(self, selected: list[dict[str, Any]], traces: dict, evaluation: dict) -> None:
        batches: dict[bool, list[dict[str, Any]]] = defaultdict(list)
        durations: dict[str, list[float]] = defaultdict(list)
        trace_errors: Counter[str] = Counter()
        for observation in selected:
            inputs = observation["inputs"]
            record = {
                "inputs": inputs if isinstance(inputs, dict) else {"request": inputs},
                "outputs": observation["outputs"],
            }
            if self.expectations is not None:
                record["expectations"] = self.expectations[observation["case_index"]]
            trace_id = observation.get("trace_id")
            if self.traces and trace_id:
                traces["correlated"] += 1
                try:
                    trace = self.client.get_trace(trace_id)
                    if trace is not None:
                        record["trace"] = trace
                        traces["fetched"] += 1
                        for span in trace.data.spans:
                            kind = span.span_type if span.span_type in _SPAN_TYPES else "UNKNOWN"
                            if span.end_time_ns is not None:
                                durations[kind].append(
                                    max(0, span.end_time_ns - span.start_time_ns) / 1e6
                                )
                        url = self._url(trace_id=trace_id)
                        if url:
                            traces["urls"].append(url)
                except Exception as exc:
                    traces["errors"] += 1
                    trace_errors[type(exc).__name__] += 1
            batches["trace" in record].append(record)
        if self.traces:
            traces.update(
                status=(
                    "no_samples"
                    if not selected
                    else "completed"
                    if traces["fetched"] == len(selected)
                    else "partial"
                ),
                missing=len(selected) - traces["fetched"],
                coverage=traces["fetched"] / len(selected) if selected else None,
                error_types=sorted(trace_errors),
                spans=[
                    {
                        "span_type": kind,
                        "count": len(values),
                        "total_duration_ms": sum(values),
                        "mean_duration_ms": sum(values) / len(values),
                    }
                    for kind, values in sorted(durations.items())
                ],
            )
        if self._scorers:
            evaluation["status"] = "no_samples"
            if selected:
                try:
                    with self.client.evaluation_run(self.experiment_id) as run_id:
                        evaluation.update(run_id=run_id, status="completed")
                        for name, scorer in zip(self.scorer_names, self._scorers, strict=True):
                            outcome = {
                                "name": name,
                                "status": "running",
                                "selected": len(selected),
                                "scored": 0,
                                "errors": 0,
                                "coverage": None,
                                "metrics": {},
                                "error_types": [],
                            }
                            evaluation["scorers"].append(outcome)
                            unknown_coverage = False
                            for has_trace, batch in batches.items():
                                try:
                                    facts = self.client.evaluate(batch, scorer)
                                    if facts.scored is None:
                                        unknown_coverage = True
                                    else:
                                        outcome["scored"] += facts.scored
                                    outcome["errors"] += facts.errors or 0
                                    for key, value in facts.metrics.items():
                                        # Medians and percentiles cannot be averaged across batches.
                                        prefix = (
                                            ("traced/" if has_trace else "untraced/")
                                            if len(batches) > 1
                                            else ""
                                        )
                                        outcome["metrics"][prefix + key] = value
                                except Exception as exc:
                                    outcome["errors"] += len(batch)
                                    outcome["error_types"].append(type(exc).__name__)
                            outcome["coverage"] = (
                                None if unknown_coverage else outcome["scored"] / len(selected)
                            )
                            outcome["status"] = "completed"
                            if outcome["scored"] < len(selected) or outcome["errors"]:
                                outcome["status"] = "partial" if outcome["scored"] else "failed"
                            if unknown_coverage:
                                outcome.update(status="partial", scored=None)
                            self.client.log_metrics(outcome["metrics"])
                        evaluation["run_url"] = self._url(run_id=run_id)
                    statuses = {item["status"] for item in evaluation["scorers"]}
                    evaluation["status"] = (
                        "failed"
                        if statuses == {"failed"}
                        else "partial"
                        if statuses != {"completed"}
                        else "completed"
                    )
                except Exception as exc:
                    evaluation.update(status="failed", error_types=[type(exc).__name__])
