"""Unit tests for the agent-side tracing helpers in `databricks_mason.runtime.tracing`."""

from __future__ import annotations

from unittest import mock

import pytest

from databricks_mason.runtime import tracing

_DEST_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACING_DESTINATION")
_EXP_VARS = ("MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_NAME")


@pytest.fixture(autouse=True)
def _reset_enabled():
    original = tracing._enabled
    yield
    tracing._enabled = original


def _clear_env(monkeypatch):
    for var in _DEST_VARS + _EXP_VARS:
        monkeypatch.delenv(var, raising=False)


def test_configure_tracing_enables_and_calls_autolog(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "123")
    autolog = mock.Mock()
    tracing.configure_tracing(autolog=autolog)
    assert tracing._enabled is True
    autolog.assert_called_once_with()  # framework autolog bound in only when enabled


def test_configure_tracing_disables_without_config(monkeypatch):
    _clear_env(monkeypatch)
    autolog = mock.Mock()
    with mock.patch.object(tracing.mlflow.tracing, "disable") as disable:
        tracing.configure_tracing(autolog=autolog)
    assert tracing._enabled is False
    autolog.assert_not_called()
    disable.assert_called_once()  # disabled outright so the per-request span has nothing to export


def test_configure_tracing_requires_both_halves(monkeypatch):
    # A destination without an experiment (or vice versa) stays disabled.
    _clear_env(monkeypatch)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    with mock.patch.object(tracing.mlflow.tracing, "disable"):
        tracing.configure_tracing(autolog=mock.Mock())
    assert tracing._enabled is False


def test_request_span_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", False)
    with mock.patch.object(tracing.mlflow, "start_span") as start:
        with tracing.request_span(name="agent", inputs={"a": 1}) as span:
            assert span is None
    start.assert_not_called()  # no span, no mlflow touched when tracing is off


def test_request_span_opens_span_and_sets_inputs_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", True)
    fake_span = mock.Mock()
    cm = mock.MagicMock()
    cm.__enter__.return_value = fake_span
    with mock.patch.object(tracing.mlflow, "start_span", return_value=cm) as start:
        with tracing.request_span(name="agent", inputs={"a": 1}) as span:
            assert span is fake_span
    start.assert_called_once_with(name="agent")
    fake_span.set_inputs.assert_called_once_with({"a": 1})


def test_request_span_skips_inputs_when_none(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", True)
    fake_span = mock.Mock()
    cm = mock.MagicMock()
    cm.__enter__.return_value = fake_span
    with mock.patch.object(tracing.mlflow, "start_span", return_value=cm):
        with tracing.request_span() as span:
            assert span is fake_span
    fake_span.set_inputs.assert_not_called()


def test_tag_session_updates_current_trace_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", True)
    with mock.patch.object(tracing.mlflow, "update_current_trace") as upd:
        tracing.tag_session("sess-1")
    upd.assert_called_once()


def test_tag_session_noop_when_disabled_or_empty(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", True)
    with mock.patch.object(tracing.mlflow, "update_current_trace") as upd:
        tracing.tag_session("")  # no session id -> nothing to tag
    upd.assert_not_called()
    monkeypatch.setattr(tracing, "_enabled", False)
    with mock.patch.object(tracing.mlflow, "update_current_trace") as upd:
        tracing.tag_session("sess-1")  # tracing off -> nothing to tag
    upd.assert_not_called()
