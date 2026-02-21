import importlib
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _reload_module():
    yield
    import databricks_openai.agents

    importlib.reload(databricks_openai.agents)


@patch("agents.tracing.set_trace_processors")
def test_agents_import_disables_tracing_by_default(mock_set_trace_processors):
    import databricks_openai.agents

    importlib.reload(databricks_openai.agents)
    mock_set_trace_processors.assert_called_with([])


@patch("agents.tracing.set_trace_processors")
def test_agents_import_keeps_tracing_when_env_var_set(
    mock_set_trace_processors, monkeypatch
):
    monkeypatch.setenv("ENABLE_OPENAI_AGENTS_TRACING", "true")
    import databricks_openai.agents

    importlib.reload(databricks_openai.agents)
    mock_set_trace_processors.assert_not_called()
