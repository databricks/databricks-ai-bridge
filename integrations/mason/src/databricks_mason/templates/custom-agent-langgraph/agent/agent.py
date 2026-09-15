from typing import Any

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent

from databricks_mason import workspace_client, workspace_headers

MODEL = "databricks-gpt-5-2"


class _RoutedChatDatabricks(ChatDatabricks):
    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


async def invoke(messages: list[dict[str, Any]]) -> dict[str, Any]:
    agent = create_agent(
        model=_RoutedChatDatabricks(endpoint=MODEL, workspace_client=workspace_client()),
        tools=[],
    )
    result = await agent.ainvoke({"messages": messages})
    message = result["messages"][-1]
    output = message.model_dump() if hasattr(message, "model_dump") else message
    return {"output": [output]}
