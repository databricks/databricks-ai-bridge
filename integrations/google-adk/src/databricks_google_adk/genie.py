from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import Genie
from google.adk.tools import FunctionTool


def create_genie_tool(
    space_id: str,
    tool_name: str = "ask_genie",
    tool_description: str | None = None,
    client: Optional[WorkspaceClient] = None,
    return_pandas: bool = False,
) -> FunctionTool:
    """
    Create a Google ADK tool that queries a Databricks Genie space.

    Genie is Databricks' AI/BI assistant that can answer natural language questions
    about your data by generating and executing SQL queries.

    Args:
        space_id: The ID of the Genie space to query.
        tool_name: Name for the tool (default: "ask_genie").
        tool_description: Custom description for the tool. If not provided,
            uses the Genie space's description.
        client: Optional WorkspaceClient instance for authentication.
        return_pandas: Whether to return results as pandas DataFrames
            (if False, returns markdown strings).

    Returns:
        A FunctionTool that can be used with Google ADK agents.

    Example:
        ```python
        from databricks_google_adk import create_genie_tool
        from google.adk.agents import Agent

        # Create the Genie tool
        genie_tool = create_genie_tool(
            space_id="your-genie-space-id",
            tool_description="Ask questions about sales data",
        )

        # Use with an ADK agent
        agent = Agent(
            name="data_analyst",
            model="gemini-2.0-flash",
            instruction="You are a data analyst. Use the genie tool to answer data questions.",
            tools=[genie_tool],
        )
        ```
    """
    import mlflow

    genie = Genie(
        space_id=space_id,
        client=client,
        return_pandas=return_pandas,
    )

    # Use space description if no custom description provided
    description = tool_description or genie.description or (
        "Ask questions about data in natural language. "
        "This tool queries a Databricks Genie space that can generate and execute SQL queries."
    )

    # Track conversation state for multi-turn conversations
    conversation_state = {"conversation_id": None}

    def ask_genie(question: str, new_conversation: bool = False) -> dict:
        """
        Ask a question to the Databricks Genie AI/BI assistant.

        Args:
            question: The natural language question to ask about the data.
            new_conversation: If True, starts a new conversation. If False,
                continues the previous conversation if one exists.

        Returns:
            A dictionary containing:
            - result: The query result (markdown table or text)
            - query: The generated SQL query (if applicable)
            - description: Explanation of the query logic
            - conversation_id: ID to continue this conversation
        """
        with mlflow.start_span(name="ask_genie", span_type="TOOL"):
            # Determine conversation_id
            conv_id = None if new_conversation else conversation_state.get("conversation_id")

            # Query Genie
            response = genie.ask_question(question, conversation_id=conv_id)

            # Update conversation state
            if response.conversation_id:
                conversation_state["conversation_id"] = response.conversation_id

            # Format result
            result = response.result
            if hasattr(result, "to_markdown"):
                # pandas DataFrame
                result = result.to_markdown(index=False)

            return {
                "result": result,
                "query": response.query or "",
                "description": response.description or "",
                "conversation_id": response.conversation_id or "",
            }

    # Set function metadata
    ask_genie.__name__ = tool_name
    ask_genie.__doc__ = description

    return FunctionTool(ask_genie)


class GenieTool:
    """
    A wrapper class for Databricks Genie that provides a Google ADK compatible tool.

    This class maintains conversation state and provides both synchronous and
    tool-based access to Genie.

    Example:
        ```python
        from databricks_google_adk import GenieTool
        from google.adk.agents import Agent

        # Create the Genie tool wrapper
        genie = GenieTool(space_id="your-genie-space-id")

        # Use with an ADK agent
        agent = Agent(
            name="data_analyst",
            model="gemini-2.0-flash",
            instruction="You are a data analyst.",
            tools=[genie.as_tool()],
        )

        # Or call directly
        result = genie.ask("What were total sales last month?")
        ```
    """

    def __init__(
        self,
        space_id: str,
        tool_name: str = "ask_genie",
        tool_description: str | None = None,
        client: Optional[WorkspaceClient] = None,
        return_pandas: bool = False,
    ):
        """
        Initialize the GenieTool.

        Args:
            space_id: The ID of the Genie space to query.
            tool_name: Name for the tool (default: "ask_genie").
            tool_description: Custom description for the tool.
            client: Optional WorkspaceClient instance.
            return_pandas: Whether to return results as pandas DataFrames.
        """
        self.space_id = space_id
        self.tool_name = tool_name
        self._client = client
        self._return_pandas = return_pandas
        self._genie = Genie(
            space_id=space_id,
            client=client,
            return_pandas=return_pandas,
        )
        self._tool_description = tool_description or self._genie.description
        self._conversation_id: str | None = None
        self._adk_tool: FunctionTool | None = None

    @property
    def description(self) -> str:
        """Get the Genie space description."""
        return self._genie.description or ""

    @property
    def conversation_id(self) -> str | None:
        """Get the current conversation ID."""
        return self._conversation_id

    def reset_conversation(self) -> None:
        """Reset the conversation state to start fresh."""
        self._conversation_id = None

    def ask(self, question: str, new_conversation: bool = False) -> dict:
        """
        Ask a question to Genie.

        Args:
            question: The natural language question to ask.
            new_conversation: If True, starts a new conversation.

        Returns:
            A dictionary with result, query, description, and conversation_id.
        """
        import mlflow

        with mlflow.start_span(name="GenieTool.ask", span_type="TOOL"):
            conv_id = None if new_conversation else self._conversation_id

            response = self._genie.ask_question(question, conversation_id=conv_id)

            if response.conversation_id:
                self._conversation_id = response.conversation_id

            result = response.result
            if hasattr(result, "to_markdown"):
                result = result.to_markdown(index=False)

            return {
                "result": result,
                "query": response.query or "",
                "description": response.description or "",
                "conversation_id": response.conversation_id or "",
            }

    def as_tool(self) -> FunctionTool:
        """
        Convert this GenieTool to a Google ADK FunctionTool.

        Returns:
            A FunctionTool that can be used with Google ADK agents.
        """
        if self._adk_tool is not None:
            return self._adk_tool

        # Create a closure that references self
        def ask_genie(question: str, new_conversation: bool = False) -> dict:
            """Ask a question to the Databricks Genie AI/BI assistant."""
            return self.ask(question, new_conversation)

        ask_genie.__name__ = self.tool_name
        ask_genie.__doc__ = self._tool_description or (
            "Ask questions about data in natural language. "
            "This tool queries a Databricks Genie space."
        )

        self._adk_tool = FunctionTool(ask_genie)
        return self._adk_tool
