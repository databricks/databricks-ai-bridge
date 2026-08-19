"""OpenAI Agents SDK app backed by DatabricksDurableRuntime."""

import os
import re
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from review_agent import execute_review, resume_review
from sessions import create_session, initialize_sessions

from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)

PR_URL = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[0-9]+/?$")
RECOVERY_NOTE = (
    "This is a crash-recovery attempt. Continue the interrupted task using only "
    "the persisted SDK session history. Inspect the shell workspace and safely "
    "repeat any interrupted tool."
)
RESPONSE_STATUS = {
    DurableExecutionStatus.QUEUED: "in_progress",
    DurableExecutionStatus.ACTIVE: "in_progress",
    DurableExecutionStatus.COMPLETED: "completed",
    DurableExecutionStatus.FAILED: "failed",
}


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
        raise HTTPException(400, "custom_inputs.pr_url must be a public GitHub pull-request URL")
    try:
        minimum_minutes = float(custom_inputs.get("minimum_minutes", 30))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "custom_inputs.minimum_minutes must be a number") from exc
    if not 0 <= minimum_minutes <= 60:
        raise HTTPException(400, "custom_inputs.minimum_minutes must be between 0 and 60")
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


async def execute_durable_review(
    request: JsonObject,
    context: DurableExecutionContext,
) -> JsonObject:
    agent_request = ResponsesAgentRequest.model_validate(request)
    session_id = _session_id(agent_request)
    pr_url, minimum_minutes = _review_inputs(agent_request)
    session = create_session(session_id)
    if context.is_recovery:
        report = await resume_review(session, RECOVERY_NOTE)
    else:
        report = await execute_review(pr_url, minimum_minutes, session)
    response = ResponsesAgentResponse.model_validate(
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
    return response.model_dump(mode="json", exclude_none=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    await initialize_sessions()
    durable_runtime = DatabricksDurableRuntime(
        execute_durable_review,
        schema=os.getenv("LAKEBASE_DURABILITY_SCHEMA", "openai_sdk_agent_durability"),
    )
    await durable_runtime.start()
    application.state.durable_runtime = durable_runtime
    try:
        yield
    finally:
        await durable_runtime.stop()


app = FastAPI(title="Durable OpenAI SDK PR review agent", lifespan=lifespan)


def _status_response(state: DurableExecution) -> ResponsesAgentResponse:
    if state.status == DurableExecutionStatus.COMPLETED:
        if state.response is None:
            raise RuntimeError(f"execution {state.execution_id!r} completed without a response")
        return ResponsesAgentResponse.model_validate(state.response)
    custom_inputs = dict(state.request.get("custom_inputs") or {})
    return ResponsesAgentResponse(
        id=state.execution_id,
        status=RESPONSE_STATUS[state.status],
        output=[],
        custom_outputs={
            "execution_id": state.execution_id,
            "session_id": custom_inputs.get("session_id", state.execution_id),
            "attempt": state.attempt,
        },
    )


@app.get("/health")
@app.get("/api/healthz")
async def health() -> dict:
    return {"ok": True, "model": "gpt-5.6-luna"}


@app.post("/responses")
@app.post("/invocations")
async def invoke(
    request: ResponsesAgentRequest,
    response: Response,
    http_request: Request,
) -> ResponsesAgentResponse:
    if request.stream:
        raise HTTPException(400, "streaming is not implemented by this example")
    _review_inputs(request)
    execution_id = _session_id(request)
    payload = _durable_request(request, execution_id)
    durable_runtime: DatabricksDurableRuntime = http_request.app.state.durable_runtime
    try:
        if bool(getattr(request, "background", False)):
            state = await durable_runtime.submit(execution_id, payload)
            if not state.is_terminal:
                response.status_code = 202
            return _status_response(state)
        result = await durable_runtime.invoke(execution_id, payload)
        return ResponsesAgentResponse.model_validate(result)
    except DurableRequestConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except DurableExecutionFailedError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/responses/{execution_id}")
async def retrieve(execution_id: str, request: Request) -> ResponsesAgentResponse:
    durable_runtime: DatabricksDurableRuntime = request.app.state.durable_runtime
    state = await durable_runtime.get(execution_id)
    if state is None:
        raise HTTPException(404, f"execution {execution_id!r} was not found")
    return _status_response(state)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("DATABRICKS_APP_PORT", "8000")))
