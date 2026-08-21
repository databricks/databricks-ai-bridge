"""OpenAI Agents SDK handlers for framework-managed recovery."""

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
from openai_sdk_agent_shared.review_agent import execute_review, stream_review
from openai_sdk_agent_shared.sessions import create_session

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


def _parse_prompt(request: ResponsesAgentRequest) -> str:
    if not request.input:
        return ""
    content = request.input[-1].model_dump().get("content")
    return content if isinstance(content, str) else ""


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


# Optional override. Without it, LongRunningAgentServer reconstructs recovery
# prose from the prior event log and rotates to a fresh SDK session. The server
# calls this once after claiming a stale attempt, then reuses the required
# stream-handler execution mode.
#
# from databricks_ai_bridge.long_running import ResumeContext, on_resume
#
# @on_resume()
# async def resume_request(
#     request: ResponsesAgentRequest,
#     context: ResumeContext,
# ) -> ResponsesAgentRequest:
#     request_dict = request.model_dump(exclude_none=True)
#     recovery_message = {
#         "type": "message",
#         "role": "user",
#         "content": (
#             "[RECOVERY] The previous attempt of this agent task crashed "
#             "mid-execution. Below is the raw stream-event log from that "
#             "attempt as JSON. Some tool calls may have completed and some "
#             "may have been interrupted before returning a result. Inspect "
#             "the events, figure out what is already done versus in-progress "
#             "/ not completed, and continue the task from where it left off. "
#             "If a tool call was interrupted, you may re-invoke it if its "
#             "result is still needed.\n\n"
#             f"Events:\n{json.dumps(list(context.previous_events))}"
#         ),
#     }
#     request_dict["input"] = [
#         *request_dict.get("input", []),
#         recovery_message,
#     ]
#
#     custom_inputs = dict(request_dict.get("custom_inputs") or {})
#     explicit_thread_id = custom_inputs.pop("thread_id", None)
#     explicit_session_id = custom_inputs.pop("session_id", None)
#     request_context = dict(request_dict.get("context") or {})
#     base_session_id = str(
#         explicit_thread_id
#         or explicit_session_id
#         or request_context.get("conversation_id")
#         or context.response_id
#     )
#     request_context["conversation_id"] = (
#         f"{base_session_id}::attempt-{context.attempt_number}"
#     )
#     request_dict["custom_inputs"] = custom_inputs
#     request_dict["context"] = request_context
#     return ResponsesAgentRequest(**request_dict)
#
# The complete implementation above can be replaced with:
#     return await context.default_request(request)
#
# Resume translation ends here. The invoke and stream handlers below do not
# detect recovery or choose a recovery-specific path; they process the
# transformed request exactly like every other request.


@invoke()
async def invoke_review(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    session = create_session(_session_id(request))
    pr_url, minimum_minutes = _review_inputs(request)
    report = await execute_review(
        pr_url,
        minimum_minutes,
        session,
        _parse_prompt(request),
    )
    return ResponsesAgentResponse(output=[_message(report)])


@stream()
async def stream_review_events(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent | dict, None]:
    session = create_session(_session_id(request))
    pr_url, minimum_minutes = _review_inputs(request)
    events = stream_review(
        pr_url,
        minimum_minutes,
        session,
        _parse_prompt(request),
    )
    async for event in _responses_events(events):
        yield event
