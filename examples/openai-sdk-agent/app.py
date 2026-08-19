"""OpenAI Agents SDK app backed by DatabricksDurableAgentServer."""

import os
import re
from uuid import uuid4

import uvicorn
from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from review_agent import execute_review, resume_review
from sessions import create_session, initialize_sessions

from databricks_ai_bridge.durable_agent_server import (
    DatabricksDurableAgentServer,
    PreparedDurableRequest,
    get_durable_execution_context,
)
from databricks_ai_bridge.durable_runtime import JsonObject

PR_URL = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+/?$")
RECOVERY_NOTE = (
    "This is a crash-recovery attempt. Continue the interrupted task using only "
    "the persisted SDK session history. Inspect the shell workspace and safely "
    "repeat any interrupted tool."
)


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


def _durable_request(request: ResponsesAgentRequest, session_id: str) -> JsonObject:
    payload = request.model_dump(mode="json", exclude_none=True)
    payload.pop("background", None)
    payload.pop("stream", None)
    custom_inputs = dict(payload.get("custom_inputs") or {})
    custom_inputs["session_id"] = session_id
    payload["custom_inputs"] = custom_inputs
    return payload


def _message(text: str) -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


@invoke()
async def invoke_review(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    context = get_durable_execution_context()
    session_id = _session_id(request)
    pr_url, minimum_minutes = _review_inputs(request)
    session = create_session(session_id)
    if context.is_recovery:
        report = await resume_review(session, RECOVERY_NOTE)
    else:
        report = await execute_review(pr_url, minimum_minutes, session)
    return ResponsesAgentResponse.model_validate(
        {
            "id": context.execution_id,
            "status": "completed",
            "output": [_message(report)],
            "custom_outputs": {
                "execution_id": context.execution_id,
                "session_id": session_id,
                "attempt": context.attempt,
            },
        }
    )


def prepare_review_request(request: ResponsesAgentRequest) -> PreparedDurableRequest:
    _review_inputs(request)
    session_id = _session_id(request)
    return PreparedDurableRequest(
        execution_id=session_id,
        payload=_durable_request(request, session_id),
    )


server = DatabricksDurableAgentServer(
    prepare_request=prepare_review_request,
    on_startup=initialize_sessions,
    schema=os.getenv("LAKEBASE_DURABILITY_SCHEMA", "openai_sdk_agent_durability"),
)
app = server.app


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("DATABRICKS_APP_PORT", "8000")))
