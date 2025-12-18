import asyncio
from typing import Annotated, Any, AsyncGenerator, Generator, Optional, Sequence, TypedDict, Union

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
)
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

nest_asyncio.apply()
############################################
## Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# TODO: Update with your system prompt
system_prompt = """
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
databricks_mcp_client = DatabricksMultiServerMCPClient(
    [
        DatabricksMCPServer(
            name="system-ai",
            url=f"{host}/api/2.0/mcp/functions/system/ai",
        ),
        # DatabricksMCPServer(
        #     name="custom_mcp",
        #     url="custom_app_url",
        #     workspace_client=custom_mcp_server_workspace_client
        # ),
        # DatabricksMCPServer(
        #     name="obo_vs_client",
        #     url=f"{host}/api/2.0/mcp/vector-search/system/ai",
        #     workspace_client=obo_workspace_client
        # )
    ]
)


# The state for the agent workflow, including the conversation and any custom data
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)  # Bind tools to the model

    # Function to check if agent should continue or finish based on last message
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If function (tool) calls are present, continue; otherwise, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # Preprocess: optionally prepend a system prompt to the conversation history
    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model  # Chain the preprocessor and the model

    # The function to invoke the model within the workflow
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)  # Create the agent's state machine

    workflow.add_node("agent", RunnableLambda(call_model))  # Agent node (LLM)
    workflow.add_node("tools", ToolNode(tools))  # Tools node

    workflow.set_entry_point("agent")  # Start at agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",  # If the model requests a tool call, move to tools node
            "end": END,  # Otherwise, end the workflow
        },
    )
    workflow.add_edge("tools", "agent")  # After tools are called, return to agent node

    # Compile and return the tool-calling agent workflow
    return workflow.compile()


# ResponsesAgent class to wrap the compiled agent and make it compatible with Mosaic AI Responses API
class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    # Make a prediction (single-step) for the agent
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done" or event.type == "error"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    async def _predict_stream_async(
        self,
        request: ResponsesAgentRequest,
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        # Stream events from the agent graph
        async for event in self.agent.astream(
            {"messages": cc_msgs}, stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                # Stream updated messages from the workflow nodes
                for node_data in event[1].values():
                    if len(node_data.get("messages", [])) > 0:
                        for item in output_to_responses_items_stream(node_data["messages"]):
                            yield item
            elif event[0] == "messages":
                # Stream generated text message chunks
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except:
                    pass

    # Stream predictions for the agent, yielding output as it's generated
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


# Initialize the entire agent, including MCP tools and workflow
def initialize_agent():
    """Initialize the agent with MCP tools"""
    # Create MCP tools from the configured servers
    mcp_tools = asyncio.run(databricks_mcp_client.get_tools())

    # Create the agent graph with an LLM, tool set, and system prompt (if given)
    agent = create_tool_calling_agent(llm, mcp_tools, system_prompt)
    return LangGraphResponsesAgent(agent)


mlflow.langchain.autolog()
AGENT = initialize_agent()
mlflow.models.set_model(AGENT)
