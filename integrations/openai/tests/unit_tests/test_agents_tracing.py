import importlib
from unittest.mock import patch


@patch("agents.tracing.set_trace_processors")
def test_import_disables_agents_tracing_by_default(mock_set_trace_processors):
    import databricks_openai

    importlib.reload(databricks_openai)
    mock_set_trace_processors.assert_called_with([])


@patch("agents.tracing.set_trace_processors")
def test_import_keeps_agents_tracing_when_env_var_set(
    mock_set_trace_processors, monkeypatch
):
    monkeypatch.setenv("ENABLE_OPENAI_AGENTS_TRACING", "true")
    import databricks_openai

    importlib.reload(databricks_openai)
    mock_set_trace_processors.assert_not_called()
