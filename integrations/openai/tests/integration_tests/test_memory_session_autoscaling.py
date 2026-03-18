"""
Integration tests for AsyncDatabricksSession with autoscaling mode.

These tests require:
1. A Lakebase autoscaling instance to be available
2. Valid Databricks authentication (DATABRICKS_HOST + DATABRICKS_CLIENT_ID/SECRET or profile)

Set at least one of these environment variables:
    LAKEBASE_PROJECT + LAKEBASE_BRANCH: Autoscaling project and branch names
    LAKEBASE_AUTOSCALING_ENDPOINT: Full autoscaling endpoint resource path

Example:
    LAKEBASE_PROJECT=my-project LAKEBASE_BRANCH=main \
        pytest tests/integration_tests/test_memory_session_autoscaling.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any, cast

import pytest

# Skip all tests if no autoscaling env vars are set
pytestmark = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_PROJECT") and not os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT"),
    reason="No Lakebase autoscaling env vars set "
    "(need LAKEBASE_PROJECT or LAKEBASE_AUTOSCALING_ENDPOINT)",
)

_skip_no_project_branch = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_PROJECT") or not os.environ.get("LAKEBASE_BRANCH"),
    reason="LAKEBASE_PROJECT and LAKEBASE_BRANCH not set",
)

_skip_no_endpoint = pytest.mark.skipif(
    not os.environ.get("LAKEBASE_AUTOSCALING_ENDPOINT"),
    reason="LAKEBASE_AUTOSCALING_ENDPOINT not set",
)


# =============================================================================
# Helpers
# =============================================================================


def _get_unique_table_names() -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    return f"test_sessions_{suffix}", f"test_messages_{suffix}"


async def _run_crud_test(conn_kwargs: dict, cleanup_tables: list):
    """Test CRUD lifecycle: empty -> add -> get -> pop -> clear."""
    from databricks_openai.agents import AsyncDatabricksSession

    sessions_table, messages_table = _get_unique_table_names()
    cleanup_tables.append((sessions_table, messages_table))

    session = AsyncDatabricksSession(
        session_id=str(uuid.uuid4()),
        sessions_table=sessions_table,
        messages_table=messages_table,
        **conn_kwargs,
    )

    # Empty session
    items = cast(list[Any], await session.get_items())
    assert items == []

    # Add and retrieve
    test_items: list[Any] = [
        {"role": "user", "content": "Hello from autoscaling"},
        {"role": "assistant", "content": "Autoscaling response"},
    ]
    await session.add_items(test_items)

    items = cast(list[Any], await session.get_items())
    assert len(items) == 2
    assert items[0]["content"] == "Hello from autoscaling"
    assert items[1]["content"] == "Autoscaling response"

    # Pop last item
    popped = cast(Any, await session.pop_item())
    assert popped is not None
    assert popped["role"] == "assistant"

    # Clear
    await session.clear_session()
    items = cast(list[Any], await session.get_items())
    assert items == []


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cleanup_tables_project_branch():
    """Track and clean up test tables on the project/branch autoscaling database."""
    tables_to_cleanup: list[tuple[str, str]] = []
    yield tables_to_cleanup
    if tables_to_cleanup:
        from databricks_ai_bridge.lakebase import LakebasePool

        pool = LakebasePool(
            project=os.environ["LAKEBASE_PROJECT"], branch=os.environ["LAKEBASE_BRANCH"]
        )
        with pool.connection() as conn:
            for sessions_table, messages_table in tables_to_cleanup:
                conn.execute(f"DROP TABLE IF EXISTS {messages_table}")
                conn.execute(f"DROP TABLE IF EXISTS {sessions_table}")
        pool.close()


@pytest.fixture
def cleanup_tables_endpoint():
    """Track and clean up test tables on the endpoint autoscaling database."""
    tables_to_cleanup: list[tuple[str, str]] = []
    yield tables_to_cleanup
    if tables_to_cleanup:
        from databricks_ai_bridge.lakebase import LakebasePool

        pool = LakebasePool(autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"])
        with pool.connection() as conn:
            for sessions_table, messages_table in tables_to_cleanup:
                conn.execute(f"DROP TABLE IF EXISTS {messages_table}")
                conn.execute(f"DROP TABLE IF EXISTS {sessions_table}")
        pool.close()


# =============================================================================
# Autoscaling Session Tests
# =============================================================================


class TestSessionAutoscaling:
    """Test AsyncDatabricksSession with autoscaling modes (project/branch and endpoint)."""

    @_skip_no_project_branch
    @pytest.mark.asyncio
    async def test_crud_project_branch(self, cleanup_tables_project_branch):
        """Test autoscaling project/branch params forwarded to AsyncLakebaseSQLAlchemy."""
        await _run_crud_test(
            {"project": os.environ["LAKEBASE_PROJECT"], "branch": os.environ["LAKEBASE_BRANCH"]},
            cleanup_tables_project_branch,
        )

    @_skip_no_endpoint
    @pytest.mark.asyncio
    async def test_crud_endpoint(self, cleanup_tables_endpoint):
        """Test endpoint autoscaling params forwarded to AsyncLakebaseSQLAlchemy."""
        await _run_crud_test(
            {"autoscaling_endpoint": os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"]},
            cleanup_tables_endpoint,
        )
