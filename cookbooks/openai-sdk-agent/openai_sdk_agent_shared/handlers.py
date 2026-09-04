"""MLflow handler registrations shared by both recovery examples."""

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
    if custom_inputs.get("session_id"):
        return str(custom_inputs["session_id"])
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    return str(uuid4())


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
