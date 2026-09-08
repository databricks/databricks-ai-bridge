import os
from typing import Any

from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

MODEL = "databricks-gpt-5-2"


def _configure_client() -> None:
    workspace_id = os.getenv("DATABRICKS_WORKSPACE_ID", "").strip()
    headers = {"X-Databricks-Org-Id": workspace_id} if workspace_id else {}
    workspace = WorkspaceClient(custom_headers=headers) if headers else WorkspaceClient()
    client = (
        AsyncDatabricksOpenAI(workspace_client=workspace, default_headers=headers)
        if headers
        else AsyncDatabricksOpenAI(workspace_client=workspace)
    )
    set_default_openai_client(client)
    set_default_openai_api("chat_completions")


async def invoke(messages: list[dict[str, Any]]) -> dict[str, Any]:
    _configure_client()
    result = await Runner.run(
        Agent(name="Agent", instructions="You are a helpful assistant.", model=MODEL),
        messages,
    )
    return {"output": [{"role": "assistant", "content": str(result.final_output)}]}
