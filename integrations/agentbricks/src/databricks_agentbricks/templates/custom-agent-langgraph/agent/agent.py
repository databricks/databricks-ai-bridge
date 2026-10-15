import os
from typing import Any

from databricks_langchain import ChatDatabricks

from databricks_agentkit.claude import prepare_claude_request
from langchain.agents import create_agent

from databricks_agentkit import workspace_client, workspace_headers

MODEL = "databricks-gpt-5-2"


class _RoutedChatDatabricks(ChatDatabricks):
    """Apply template Claude controls and forward workspace routing."""

    def _prepare_inputs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return prepare_claude_request(
            super()._prepare_inputs(*args, **kwargs),
            os.getenv("DATABRICKS_CLAUDE_PROMPT_CACHING", "auto"),
        )

    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


async def invoke(messages: list[dict[str, Any]]) -> dict[str, Any]:
    agent = create_agent(
        model=_RoutedChatDatabricks(
            endpoint=MODEL,
            workspace_client=workspace_client(),
        ),
        tools=[],
    )
    result = await agent.ainvoke({"messages": messages})
    message = result["messages"][-1]
    output = message.model_dump() if hasattr(message, "model_dump") else message
    return {"output": [output]}
