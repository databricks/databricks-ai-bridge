import asyncio
from typing import AsyncGenerator, AsyncIterator, Generator
from uuid import uuid4

import mlflow
import nest_asyncio
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.mcp import MCPServerStreamableHttpParams
from agents.result import StreamEvent
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents.mcp_server import McpServer

nest_asyncio.apply()

# Setup mlflow tracing
mlflow.openai.autolog()
# Configure the Databricks OpenAI client and set it to use chat completions in order to access foundation models
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])


############################################
## Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"

# TODO: Update with your system prompt
SYSTEM_PROMPT = """
You are a helpful assistant that can run Python code.
"""

# TODO: Choose your MCP server connection type and setup the Workspace Clients for Authentication

# ---------------------------------------------------------------------------
# Managed MCP Server — simplest setup
# ---------------------------------------------------------------------------
# Databricks manages this connection automatically using your workspace settings
# and Personal Access Token (PAT) authentication.

workspace_client = WorkspaceClient()

host = workspace_client.config.host

# ---------------------------------------------------------------------------
# Custom MCP Server — hosted as a Databricks App
# ---------------------------------------------------------------------------
# Use this if you’re running your own MCP server in Databricks.
# These require OAuth with a service principal for machine-to-machine (M2M) auth.
#
# Follow the insturctions here in order to create a SP, grant the SP query permissions on your app and then mint a client id and # secret. https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m
#
# Uncomment and fill in the settings below to use a custom MCP server.
#
# import os
# custom_mcp_server_workspace_client = WorkspaceClient(
#     host="<DATABRICKS_WORKSPACE_URL>",
#     client_id=os.getenv("DATABRICKS_CLIENT_ID"),
#     client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
#     auth_type="oauth-m2m",  # Enables service principal authentication
# )

# ---------------------------------------------------------------------------
# OBO Setup
# ---------------------------------------------------------------------------
# In order to use OBO, uncomment the code below and pass this workspace client to the appropriate McpServer below
#
# from databricks_ai_bridge import ModelServingUserCredentials
# obo_workspace_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())


class MCPToolCallingAgent(ResponsesAgent):
    async def process_agent_stream_events(
        self, async_stream: AsyncIterator[StreamEvent]
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        curr_item_id = str(uuid4())
        async for event in async_stream:
            if event.type == "raw_response_event":
                event_data = event.data.model_dump()
                if (
                    event_data["type"] == "response.output_item.done"
                    and event_data.get("item") is not None
                    and event_data["item"].get("id") is not None
                ):
                    event_data["item"]["id"] = curr_item_id
                    yield ResponsesAgentStreamEvent(**event_data)
                elif (
                    event_data["type"] == "response.output_text.delta"
                    and event_data.get("item_id") is not None
                ):
                    event_data["item_id"] = curr_item_id
                    yield ResponsesAgentStreamEvent(**event_data)
            elif (
                event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item"
            ):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=event.item.to_input_item(),
                )

    async def _predict_stream_async(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        ###############################################################################
        ## Configure MCP Servers for your agent
        ##
        ## This section sets up server connections so your agent can retrieve data or take actions.

        ## There are three connection types:
        ## 1. Managed MCP servers — fully managed by Databricks (no setup required)
        ## 2. External MCP servers — hosted outside Databricks but proxied through a
        ##    Managed MCP server proxy (some setup required)
        ## 3. Custom MCP servers — MCP servers hosted as Databricks Apps (OAuth setup required)
        ##
        ###############################################################################
        async with (
            McpServer(
                name="system-ai",
                params=MCPServerStreamableHttpParams(
                    url=f"{host}/api/2.0/mcp/functions/system/ai",
                ),
            ) as system_functions,
            McpServer(
                name="custom_app",
                params=MCPServerStreamableHttpParams(url="custom_app_url"),
                workspace_client=custom_mcp_server_workspace_client,
            ) as custom_mcp_server,
            McpServer(
                name="vs_obo_mcp",
                params=MCPServerStreamableHttpParams(
                    url=f"{host}/api/2.0/mcp/vector-search/system/ai"
                ),
                workspace_client=obo_workspace_client,
            ) as vs_obo_mcp,
        ):
            agent = Agent(
                name="agent",
                instructions=SYSTEM_PROMPT,
                model=LLM_ENDPOINT_NAME,
                mcp_servers=[
                    system_functions,
                ],
            )
            messages = [i.model_dump() for i in request.input]
            result = Runner.run_streamed(agent, input=messages)
            async for event in self.process_agent_stream_events(result.stream_events()):
                yield event

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        agen = self._predict_stream_async(request)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        ait = agen.__aiter__()

        while True:
            try:
                item = loop.run_until_complete(ait.__anext__())
            except StopAsyncIteration:
                break
            else:
                yield item


mlflow.openai.autolog()
AGENT = MCPToolCallingAgent()
mlflow.models.set_model(AGENT)
