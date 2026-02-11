"""Tests that core imports work without memory extras (psycopg, psycopg_pool).

These tests run in a CI job that does NOT install databricks-openai[memory],
ensuring that non-memory components like McpServer are importable without
pulling in psycopg / psycopg_pool / sqlalchemy.

Regression test for NameError when importing McpServer without memory extras.
"""

import pytest


def test_mcpserver_import_without_memory_extras():
    """McpServer should be importable without memory extras."""
    from databricks_openai.agents import McpServer

    assert McpServer is not None


def test_async_databricks_session_import_without_memory_extras():
    """AsyncDatabricksSession class definition should not crash without memory extras."""
    from databricks_openai.agents import AsyncDatabricksSession

    assert AsyncDatabricksSession is not None


def test_async_databricks_session_raises_helpful_error_without_memory():
    """Instantiating AsyncDatabricksSession without memory extras should raise a helpful ImportError."""
    from databricks_openai.agents import AsyncDatabricksSession

    with pytest.raises(ImportError, match="databricks-openai\\[memory\\]"):
        AsyncDatabricksSession(session_id="test", instance_name="test")
