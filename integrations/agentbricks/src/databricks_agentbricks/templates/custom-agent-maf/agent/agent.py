from typing import Any

from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient
from databricks_openai import AsyncDatabricksOpenAI

from databricks_agentkit import workspace_client, workspace_headers


async def invoke(messages: list[dict[str, Any]]) -> dict[str, Any]:
    agent = Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        client=OpenAIChatClient(
            model="system.ai.claude-sonnet-4-5",
            async_client=AsyncDatabricksOpenAI(
                workspace_client=workspace_client(),
                default_headers=workspace_headers() or None,
                use_ai_gateway=True,
            ),
        ),
    )
    result = await agent.run(
        [Message(item["role"], [item["content"]]) for item in messages]
    )
    return {"output": [{"role": "assistant", "content": result.text}]}
