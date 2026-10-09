"""Service and target-selection tests without Click or live MLflow."""

from __future__ import annotations

import pathlib
import types
from contextlib import contextmanager
from unittest import mock

import pytest

from databricks_agentbricks.clients.local_tracing_client import (
    LocalTracingClient,
    LocalTracingStart,
)
from databricks_agentbricks.clients.mlflow_trace_client import MLflowTraceClient
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.services.tracing.targets import TraceReadTarget, TraceTargetResolver
from databricks_agentbricks.services.tracing_service import TraceReadRequest, TracingService

_AGENT_TOML = 'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "agentbricks"\n'


def test_bind_by_name_does_not_construct_mlflow_client(tmp_path: pathlib.Path):
    (tmp_path / "agent.toml").write_text(_AGENT_TOML)
    mlflow = mock.create_autospec(MLflowTraceClient, instance=True)
    service = TracingService(mlflow, mock.create_autospec(TraceTargetResolver, instance=True))

    name = service.bind(tmp_path, "/Shared/agentbricks_traces/demo", None)

    assert name == "/Shared/agentbricks_traces/demo"
    assert AgentProject.load(tmp_path).trace_experiment_name == name
    mlflow.select_workspace.assert_not_called()
    mlflow.experiment_by_name.assert_not_called()
    mlflow.experiment_by_id.assert_not_called()


def test_list_returns_typed_trace_facts_from_injected_target(tmp_path: pathlib.Path):
    mlflow = mock.create_autospec(MLflowTraceClient, instance=True)
    mlflow.resolve_warehouse_id.return_value = "warehouse-1"
    mlflow.search_traces.return_value = [
        types.SimpleNamespace(
            info=types.SimpleNamespace(
                trace_id="trace-1", status="OK", execution_time_ms=12, timestamp_ms=123
            )
        )
    ]
    calls = []

    @contextmanager
    def resolve(source, experiment_name, experiment_id, warehouse_id):
        calls.append((source, experiment_name, experiment_id, warehouse_id))
        yield TraceReadTarget("databricks://DEFAULT", "experiment-1")

    targets = mock.create_autospec(TraceTargetResolver, instance=True)
    targets.resolve.side_effect = resolve

    result = TracingService(mlflow, targets).list(
        TraceReadRequest(tmp_path, warehouse_id="warehouse-1"), 7
    )

    assert calls == [(tmp_path, None, None, "warehouse-1")]
    mlflow.search_traces.assert_called_once_with("experiment-1", 7)
    assert result.experiment_id == "experiment-1"
    assert result.traces[0].trace_id == "trace-1"
    assert result.traces[0].execution_time_ms == 12


def test_get_preserves_requested_id_for_text_output(tmp_path: pathlib.Path):
    mlflow = mock.create_autospec(MLflowTraceClient, instance=True)
    mlflow.get_trace.return_value = types.SimpleNamespace(
        info=types.SimpleNamespace(trace_id="returned-id", status="OK"),
        data=types.SimpleNamespace(spans=[]),
    )

    @contextmanager
    def resolve(*args):
        yield TraceReadTarget("databricks://DEFAULT", "experiment-1")

    targets = mock.create_autospec(TraceTargetResolver, instance=True)
    targets.resolve.side_effect = resolve

    result = TracingService(mlflow, targets).get(TraceReadRequest(tmp_path), "requested-id")

    assert result.trace_id == "requested-id"
    assert result.detail.summary.trace_id == "returned-id"
    mlflow.get_trace.assert_called_once_with("requested-id")


def test_local_target_closes_server_when_mlflow_lookup_fails(tmp_path: pathlib.Path):
    (tmp_path / ".agentbricks").mkdir()
    (tmp_path / ".agentbricks" / "mlflow.db").write_text("")
    server = mock.Mock()
    mlflow = mock.create_autospec(MLflowTraceClient, instance=True)
    mlflow.experiment_by_name.side_effect = RuntimeError("lookup failed")
    local = mock.create_autospec(LocalTracingClient, instance=True)
    local.start_read.return_value = LocalTracingStart(server, "http://127.0.0.1:5599")

    with pytest.raises(RuntimeError, match="lookup failed"):
        with TraceTargetResolver(mlflow, local).resolve(tmp_path, None, None, None):
            pass

    local.stop.assert_called_once_with(server)


def test_no_bound_experiment_or_local_store_does_not_open_workspace(tmp_path: pathlib.Path):
    mlflow = mock.create_autospec(MLflowTraceClient, instance=True)
    local = mock.create_autospec(LocalTracingClient, instance=True)

    with TraceTargetResolver(mlflow, local).resolve(tmp_path, None, None, None) as target:
        assert target == TraceReadTarget(None, None)

    mlflow.select_workspace.assert_not_called()
    local.start_read.assert_not_called()
