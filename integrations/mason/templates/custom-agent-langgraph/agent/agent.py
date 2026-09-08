import os
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent

MODEL = "databricks-gpt-5-2"


class _RoutedChatDatabricks(ChatDatabricks):
    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if workspace_id := os.getenv("DATABRICKS_WORKSPACE_ID", "").strip():
            kwargs["default_headers"] = {"X-Databricks-Org-Id": workspace_id}
        return kwargs


def _workspace_client() -> WorkspaceClient:
    workspace_id = os.getenv("DATABRICKS_WORKSPACE_ID", "").strip()
    if workspace_id:
        return WorkspaceClient(custom_headers={"X-Databricks-Org-Id": workspace_id})
    return WorkspaceClient()


async def invoke(messages: list[dict[str, Any]]) -> dict[str, Any]:
    agent = create_agent(
        model=_RoutedChatDatabricks(endpoint=MODEL, workspace_client=_workspace_client()),
        tools=[],
    )
    result = await agent.ainvoke({"messages": messages})
    message = result["messages"][-1]
    output = message.model_dump() if hasattr(message, "model_dump") else message
    return {"output": [output]}
