"""Shared MLflow handlers for all three durable-server examples."""

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from uuid import uuid4

from agents.result import StreamEvent
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from shared.review_agent import execute_review, resume_review, stream_resume, stream_review
from shared.sessions import create_session

PR_URL = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+/?$")


def _session_id(request: ResponsesAgentRequest) -> str:
    custom_inputs = dict(request.custom_inputs or {})
    if custom_inputs.get("session_id"):
        return str(custom_inputs["session_id"])
    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)
    return str(uuid4())


def _review_inputs(request: ResponsesAgentRequest) -> tuple[str, float]:
    custom_inputs = dict(request.custom_inputs or {})
    pr_url = str(custom_inputs.get("pr_url") or "")
    if not PR_URL.fullmatch(pr_url):
        raise ValueError("custom_inputs.pr_url must be a public GitHub pull-request URL")
    try:
        minimum_minutes = float(custom_inputs.get("minimum_minutes", 30))
    except (TypeError, ValueError) as exc:
        raise ValueError("custom_inputs.minimum_minutes must be a number") from exc
    if not 0 <= minimum_minutes <= 60:
        raise ValueError("custom_inputs.minimum_minutes must be between 0 and 60")
    return pr_url, minimum_minutes


def _recovery_prompt(request: ResponsesAgentRequest) -> tuple[str, bool]:
    if not request.input:
        return "", False
    content = request.input[-1].model_dump().get("content")
    if not isinstance(content, str) or not content.startswith("[RECOVERY]"):
        return "", False
    return content, len(request.input) == 1 and "session store" in content


def _message(text: str) -> dict:
    return {
        "id": str(uuid4()),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


async def _responses_events(
    events: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict, None]:
    current_item_id = str(uuid4())
    async for event in events:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            if event_data["type"] == "response.output_item.added":
                current_item_id = str(uuid4())
                event_data["item"]["id"] = current_item_id
            elif isinstance(event_data.get("item"), dict) and event_data["item"].get("id"):
                event_data["item"]["id"] = current_item_id
            elif event_data.get("item_id") is not None:
                event_data["item_id"] = current_item_id
            yield event_data
        elif event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item":
            output = event.item.to_input_item()
            if not isinstance(output.get("output"), str):
                try:
                    output["output"] = json.dumps(output.get("output"))
                except (TypeError, ValueError):
                    output["output"] = str(output.get("output"))
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=output)


@invoke()
async def invoke_review(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    session = create_session(_session_id(request))
    recovery_prompt, agent_managed = _recovery_prompt(request)
    if agent_managed:
        report = await resume_review(session, recovery_prompt)
    else:
        pr_url, minimum_minutes = _review_inputs(request)
        report = await execute_review(pr_url, minimum_minutes, session, recovery_prompt)
    return ResponsesAgentResponse(output=[_message(report)])


@stream()
async def stream_review_events(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict, None]:
    session = create_session(_session_id(request))
    recovery_prompt, agent_managed = _recovery_prompt(request)
    if agent_managed:
        events = stream_resume(session, recovery_prompt)
    else:
        pr_url, minimum_minutes = _review_inputs(request)
        events = stream_review(pr_url, minimum_minutes, session, recovery_prompt)
    async for event in _responses_events(events):
        yield event
