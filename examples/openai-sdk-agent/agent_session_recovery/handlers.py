"""MLflow handlers for agent-session recovery."""

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai_sdk_agent_shared.agent import invoke_agent, stream_agent
from openai_sdk_agent_shared.sessions import create_session


def _session_id(request: ResponsesAgentRequest) -> str:
    custom_inputs = dict(request.custom_inputs or {})
    if custom_inputs.get("thread_id"):
        return str(custom_inputs["thread_id"])
    if custom_inputs.get("session_id"):
        return str(custom_inputs["session_id"])
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    return str(uuid4())


# Optional override, called only after a stale attempt is claimed. By default,
# the server keeps this session ID and sends a recovery prompt; the OpenAI
# Agents SDK then reloads its transcript. An override can modify that request:
#
# from databricks_ai_bridge.long_running import ResumeContext, on_resume
#
# @on_resume()
# async def resume(request, context: ResumeContext):
#     return await context.default_resume_request(request)


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    output = await invoke_agent(request, create_session(_session_id(request)))
    return ResponsesAgentResponse(output=output)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict[str, Any], None]:
    async for event in stream_agent(request, create_session(_session_id(request))):
        yield event
