"""Offline checks of the generated agent's native tools and history provider."""

from uuid import uuid4

import pytest
from agent_framework import Message

from agent.tools import all_tools
from databricks_agentkit.maf.sessions import DatabricksHistoryProvider


def test_local_tool_discovery():
    assert "get_current_time" in {tool.name for tool in all_tools()}


@pytest.mark.asyncio
async def test_unbound_native_history_retains_a_conversation(monkeypatch):
    monkeypatch.setattr(
        "databricks_agentkit.maf.sessions.resolve_session_store", lambda _: None
    )
    session_id = str(uuid4())
    await DatabricksHistoryProvider("alice").save_messages(
        session_id, [Message("user", ["hello"])]
    )
    history = await DatabricksHistoryProvider("alice").get_messages(session_id)
    assert history[0].text == "hello"
    assert await DatabricksHistoryProvider("bob").get_messages(session_id) == []
