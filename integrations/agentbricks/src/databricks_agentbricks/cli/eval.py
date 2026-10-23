"""Evaluate saved responses independently of endpoint invocation or load generation."""

from __future__ import annotations

import json
import signal
from pathlib import Path

import click

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.evaluation import render_evaluation_summary
from databricks_agentbricks.services.evaluation_service import (
    EvaluationService,
    load_evaluation_data,
    read_evaluation_json,
)


@click.group("eval")
def evaluation() -> None:
    """Evaluate saved agent responses and inspect their MLflow traces."""


def build_evaluation_service(obj, experiment_id: str, **options) -> EvaluationService:
    return EvaluationService(obj.profile, experiment_id, **options)


@evaluation.command("run")
@click.option(
    "--data",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Private saved-response JSON artifact produced by endpoint loadtest --save-responses.",
)
@click.option(
    "--experiment-id",
    required=True,
    help="MLflow experiment containing registered scorers and traces.",
)
@click.option(
    "--scorer",
    "scorer_names",
    multiple=True,
    help="Registered MLflow scorer name; repeat for multiple scorers.",
)
@click.option(
    "--traces", is_flag=True, help="Fetch explicit trace IDs and summarize span-type timings."
)
@click.option(
    "--expectations",
    type=click.Path(path_type=Path, dir_okay=False),
    help="JSON array of expectation objects, one per original request case.",
)
@click.option(
    "--warehouse",
    "warehouse_id",
    default=None,
    help="SQL warehouse id for reading UC-backed traces; falls back to MLFLOW_TRACING_SQL_WAREHOUSE_ID.",
)
@click.option(
    "--max-samples",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Maximum saved responses to analyze.",
)
@click.option(
    "--report-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="New or empty directory for the HTML report and summary.json.",
)
@click.pass_obj
@click.pass_context
def evaluation_run(
    ctx,
    obj,
    data,
    experiment_id,
    scorer_names,
    traces,
    expectations,
    warehouse_id,
    max_samples,
    report_dir,
) -> None:
    """Score captured answers with MLflow without invoking the target agent.

    Install databricks-agentbricks[evaluation] to use registered scorers. Trace-only
    analysis uses the base package. Evaluation stores captured inputs and outputs in
    the selected MLflow experiment; shareable reports omit their contents.
    """
    if not scorer_names and not traces:
        raise AgentCliError("Pass at least one --scorer or --traces.")
    if expectations and not scorer_names:
        raise AgentCliError("--expectations requires --scorer.")
    if warehouse_id and not traces:
        raise AgentCliError("--warehouse requires --traces.")
    case_count, observations = load_evaluation_data(data, max_samples=max_samples)
    expected = read_evaluation_json(expectations) if expectations else None
    if expectations is not None and (
        not isinstance(expected, list)
        or len(expected) != case_count
        or not all(isinstance(item, dict) for item in expected)
    ):
        raise AgentCliError(
            "Expectations must be a JSON array of objects, one per original request case."
        )
    if report_dir is not None:
        try:
            if report_dir.is_symlink() or (report_dir.exists() and any(report_dir.iterdir())):
                raise AgentCliError("--report-dir must be a new or empty directory.")
            report_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AgentCliError("Cannot create the evaluation report directory.") from exc
    service = build_evaluation_service(
        obj,
        experiment_id,
        scorer_names=scorer_names,
        traces=traces,
        max_samples=max_samples,
        expectations=expected,
        warehouse_id=warehouse_id,
    )
    caught_signal = signal.SIGINT
    previous = {}

    def interrupted(signum, frame):
        nonlocal caught_signal
        caught_signal = signum
        raise KeyboardInterrupt

    summary = {
        "schema_version": 1,
        "status": "failed",
        "experiment_id": experiment_id,
        "sample_limit": max_samples,
        "sampled_successes": len(observations),
        "evaluation": {"status": "not_run" if scorer_names else "disabled", "scorers": []},
        "traces": {"status": "not_run" if traces else "disabled"},
    }
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        try:
            summary = service.finish(observations)
        except AgentCliError as exc:
            # Detailed causes may contain captured content; only expose the exception category.
            summary["error_types"] = [type(exc.__cause__ or exc).__name__]
            click.echo(exc.message, err=True)
            if exc.hint:
                click.echo(f"  {exc.hint}", err=True)
        except KeyboardInterrupt:
            summary["status"] = "cancelled"
        except Exception as exc:
            summary["error_types"] = [type(exc).__name__]
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    try:
        if report_dir is not None:
            (report_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        render_evaluation_summary(summary, report_dir=report_dir)
    except OSError as exc:
        raise AgentCliError("Cannot write the evaluation report.") from exc
    if obj.output == "json":
        click.echo(json.dumps(summary, indent=2))
    if summary["status"] == "cancelled":
        ctx.exit(128 + caught_signal)
    if summary["status"] == "failed":
        ctx.exit(1)
