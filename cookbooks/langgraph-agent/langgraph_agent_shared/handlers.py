"""MLflow handler registrations shared by both LangGraph recovery examples."""

from collections.abc import AsyncGenerator

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from databricks_ai_bridge.long_running import (
    ResumeContext,
    ResumeStrategy,
    on_resume,
)
from langgraph_agent_shared.agent import RESUME_FROM_CHECKPOINT_KEY, stream_agent
from langgraph_agent_shared.runtime import get_checkpointer


@on_resume()
async def resume_handler(
    request: ResponsesAgentRequest,
    context: ResumeContext,
) -> ResponsesAgentRequest:
    resumed = await context.default_resume_request(request)
    if context.resume_strategy is not ResumeStrategy.AGENT_SESSION:
        return resumed

    custom_inputs = dict(resumed.custom_inputs or {})
    custom_inputs[RESUME_FROM_CHECKPOINT_KEY] = True
    return resumed.model_copy(update={"custom_inputs": custom_inputs})


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    output = [
        event.item
        async for event in stream_agent(request, get_checkpointer())
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=output)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    async for event in stream_agent(request, get_checkpointer()):
        yield event
