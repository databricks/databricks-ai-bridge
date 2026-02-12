"""
Integration tests for LangChain GenieAgent wrapper.

Tests GenieAgent initialization, execution, include_context mode,
pandas mode, conversation continuity, and auth paths against a live
Genie Space.

Prerequisites:
- Genie Space must be pre-created with test table and SP permissions
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GENIE_INTEGRATION_TESTS") != "1",
    reason="Genie integration tests disabled. Set RUN_GENIE_INTEGRATION_TESTS=1 to enable.",
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def genie_space_id():
    """Get the Genie Space ID from the GENIE_SPACE_ID environment variable."""
    space_id = os.environ.get("GENIE_SPACE_ID")
    if not space_id:
        pytest.skip("GENIE_SPACE_ID environment variable not set")
    return space_id


@pytest.fixture(scope="session")
def workspace_client():
    """Create a WorkspaceClient, converting CLI auth to PAT if needed."""
    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient()
    # Genie SDK uses WorkspaceClient directly, but we need PAT-compatible
    # auth for consistency with how the LangChain integration tests work.
    if wc.config.auth_type not in ("pat", "oauth-m2m", "model_serving_user_credentials"):
        headers = wc.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            return WorkspaceClient(host=wc.config.host, token=token, auth_type="pat")
    return wc


@pytest.fixture(scope="session")
def genie_agent(workspace_client, genie_space_id):
    """Create a GenieAgent (default string mode, no context)."""
    from databricks_langchain.genie import GenieAgent

    return GenieAgent(
        genie_space_id=genie_space_id,
        client=workspace_client,
    )


@pytest.fixture(scope="session")
def genie_agent_with_context(workspace_client, genie_space_id):
    """Create a GenieAgent with include_context=True."""
    from databricks_langchain.genie import GenieAgent

    return GenieAgent(
        genie_space_id=genie_space_id,
        include_context=True,
        client=workspace_client,
    )


@pytest.fixture(scope="session")
def genie_agent_pandas(workspace_client, genie_space_id):
    """Create a GenieAgent with return_pandas=True."""
    from databricks_langchain.genie import GenieAgent

    return GenieAgent(
        genie_space_id=genie_space_id,
        return_pandas=True,
        client=workspace_client,
    )


@pytest.fixture(scope="session")
def agent_response(genie_agent):
    """Cached agent response for 'What is the total amount by region?'"""
    return genie_agent.invoke(
        {"messages": [{"role": "user", "content": "What is the total amount by region?"}]}
    )


@pytest.fixture(scope="session")
def agent_context_response(genie_agent_with_context):
    """Cached agent response with include_context=True."""
    return genie_agent_with_context.invoke(
        {"messages": [{"role": "user", "content": "What is the total amount by region?"}]}
    )


@pytest.fixture(scope="session")
def agent_pandas_response(genie_agent_pandas):
    """Cached agent response with return_pandas=True."""
    return genie_agent_pandas.invoke(
        {"messages": [{"role": "user", "content": "What is the average amount by status?"}]}
    )


@pytest.fixture(scope="session")
def agent_continued_response(genie_agent, agent_response):
    """Cached follow-up response using existing conversation_id."""
    return genie_agent.invoke(
        {
            "messages": [{"role": "user", "content": "Now filter to only completed orders"}],
            "conversation_id": agent_response["conversation_id"],
        }
    )


# =============================================================================
# GenieAgent Initialization Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentInit:
    """Test GenieAgent initialization."""

    def test_agent_default_name(self, genie_agent):
        assert genie_agent.name == "Genie"

    def test_agent_custom_name(self, workspace_client, genie_space_id):
        from databricks_langchain.genie import GenieAgent

        agent = GenieAgent(
            genie_space_id=genie_space_id,
            genie_agent_name="CustomGenie",
            client=workspace_client,
        )
        assert agent.name == "CustomGenie"

    def test_agent_raises_on_empty_space_id(self):
        from databricks_langchain.genie import GenieAgent

        with pytest.raises(ValueError, match="genie_space_id is required"):
            GenieAgent(genie_space_id="")


# =============================================================================
# GenieAgent Execution Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentExecution:
    """Test GenieAgent invoke returns expected structure."""

    def test_response_structure(self, agent_response):
        # Our _query_genie_as_agent returns dict with messages + conversation_id
        assert isinstance(agent_response, dict)
        assert "messages" in agent_response
        assert "conversation_id" in agent_response
        assert isinstance(agent_response["conversation_id"], str)
        assert len(agent_response["conversation_id"]) > 0

    def test_messages_are_ai_messages(self, agent_response):
        from langchain_core.messages import AIMessage

        messages = agent_response["messages"]
        assert isinstance(messages, list)
        assert len(messages) > 0
        for msg in messages:
            assert isinstance(msg, AIMessage)
        # Last message (query_result) should have content
        assert len(messages[-1].content) > 0


# =============================================================================
# GenieAgent Include Context Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentIncludeContext:
    """Test GenieAgent with include_context=True."""

    def test_includes_reasoning_sql_and_result(self, agent_context_response):
        # Our _query_genie_as_agent with include_context=True adds 3 named messages
        messages = agent_context_response["messages"]
        assert len(messages) == 3
        names = [msg.name for msg in messages]
        assert names == ["query_reasoning", "query_sql", "query_result"]
        # Result message should have content
        result_msg = next(m for m in messages if m.name == "query_result")
        assert len(result_msg.content) > 0

    def test_query_sql_has_content(self, agent_context_response):
        messages = agent_context_response["messages"]
        sql_msg = next(m for m in messages if m.name == "query_sql")
        # SQL content should be a string (may be empty if Genie didn't generate SQL)
        assert isinstance(sql_msg.content, str)


# =============================================================================
# GenieAgent Pandas Mode Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentPandasMode:
    """Test GenieAgent with return_pandas=True."""

    def test_has_dataframe_key(self, agent_pandas_response):
        assert "dataframe" in agent_pandas_response

    def test_dataframe_is_pandas(self, agent_pandas_response):
        df = agent_pandas_response["dataframe"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_message_is_markdown(self, agent_pandas_response):
        messages = agent_pandas_response["messages"]
        result_msg = messages[-1]
        # When return_pandas=True, the message content is markdown of the DataFrame
        assert isinstance(result_msg.content, str)
        assert len(result_msg.content) > 0


# =============================================================================
# GenieAgent Conversation Continuity Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentConversationContinuity:
    """Test follow-up queries with conversation_id."""

    def test_continued_response_has_result(self, agent_continued_response):
        assert isinstance(agent_continued_response, dict)
        assert "messages" in agent_continued_response
        messages = agent_continued_response["messages"]
        assert len(messages) > 0
        assert len(messages[-1].content) > 0

    def test_continued_response_has_conversation_id(self, agent_continued_response):
        assert "conversation_id" in agent_continued_response
        assert isinstance(agent_continued_response["conversation_id"], str)
        assert len(agent_continued_response["conversation_id"]) > 0


# =============================================================================
# GenieAgent Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestGenieAgentAuthPaths:
    """Verify auth credentials are correctly forwarded through GenieAgent -> Genie -> WorkspaceClient."""

    def test_current_auth_produces_working_agent(self, workspace_client, genie_space_id):
        from databricks_langchain.genie import GenieAgent

        # E2E: create agent with current auth, verify it can query
        agent = GenieAgent(genie_space_id=genie_space_id, client=workspace_client)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "How many orders are there?"}]}
        )
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_pat_auth_produces_working_agent(self, workspace_client, genie_space_id):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain.genie import GenieAgent

        # Extract a bearer token from the current auth (works for any auth type)
        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        agent = GenieAgent(genie_space_id=genie_space_id, client=pat_wc)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "How many orders are there?"}]}
        )
        assert "messages" in result

    def test_oauth_m2m_auth_produces_working_agent(self, genie_space_id):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain.genie import GenieAgent

        # Skip if OAuth creds not available
        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        host = os.environ.get("DATABRICKS_HOST")
        if not (client_id and client_secret and host):
            pytest.skip(
                "OAuth-M2M credentials not available "
                "(need DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET)"
            )

        oauth_wc = WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
        )
        agent = GenieAgent(genie_space_id=genie_space_id, client=oauth_wc)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "How many orders are there?"}]}
        )
        assert "messages" in result
