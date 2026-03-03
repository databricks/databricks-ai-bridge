from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    get_user_workspace_client,
    process_agent_stream_events,
)

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])
mlflow.openai.autolog()

NAME = "agent-obo-test"
SYSTEM_PROMPT = (
    "You are a helpful assistant. When asked who the user is, "
    "call the whoami tool and return the raw result."
)
MODEL = "databricks-claude-sonnet-4-6"


def _make_whoami_tool(user_wc):
    """Create a whoami tool that uses the given workspace client."""

    @function_tool
    def whoami() -> str:
        """Returns the identity of the current user."""
        me = user_wc.current_user.me()
        return me.display_name or me.user_name or str(me.id)

    return whoami


def create_agent(tools) -> Agent:
    return Agent(
        name=NAME,
        instructions=SYSTEM_PROMPT,
        model=MODEL,
        tools=tools,
    )


@invoke()
async def invoke(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    user_wc = get_user_workspace_client()
    whoami_tool = _make_whoami_tool(user_wc)
    agent = create_agent([whoami_tool])
    messages = [i.model_dump() for i in request.input]
    result = await Runner.run(agent, messages)
    return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream(request: dict) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    user_wc = get_user_workspace_client()
    whoami_tool = _make_whoami_tool(user_wc)
    agent = create_agent([whoami_tool])
    messages = [i.model_dump() for i in request.input]
    result = Runner.run_streamed(agent, input=messages)

    async for event in process_agent_stream_events(result.stream_events()):
        yield event
