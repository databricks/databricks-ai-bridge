"""Developer-owned FastAPI wiring for DatabricksDurableRuntime."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import run_agent
from databricks_ai_bridge.durable_runtime import (
    DatabricksDurableRuntime,
    DurableExecution,
    DurableExecutionStatus,
)

runtime = DatabricksDurableRuntime(run_agent)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(lifespan=lifespan)


class RunSubmission(BaseModel):
    run_id: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    background: bool = True
    stream: bool = False


def _state_payload(state: DurableExecution) -> dict[str, Any]:
    return {
        "run_id": state.execution_id,
        "status": state.status.value,
        "attempt": state.attempt,
        "result": state.response,
    }


@app.post("/runs")
async def submit_run(submission: RunSubmission) -> Response:
    request = {"session_id": submission.session_id, "payload": submission.payload}
    if submission.stream:
        await runtime.submit(submission.run_id, request)
        return StreamingResponse(
            _event_stream(submission.run_id, 0),
            media_type="text/event-stream",
            headers={"Databricks-Run-Id": submission.run_id},
        )

    if submission.background:
        state = await runtime.submit(submission.run_id, request)
        status_code = 200 if state.is_terminal else 202
        return JSONResponse(_state_payload(state), status_code=status_code)

    await runtime.invoke(submission.run_id, request)
    state = await runtime.get(submission.run_id)
    if state is None:
        raise RuntimeError(f"run {submission.run_id!r} disappeared after completion")
    return JSONResponse(_state_payload(state))


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    state = await runtime.get(run_id)
    if state is None:
        raise HTTPException(404, "run not found")
    return _state_payload(state)


@app.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, after: int = 0) -> StreamingResponse:
    if await runtime.get(run_id) is None:
        raise HTTPException(404, "run not found")

    return StreamingResponse(
        _event_stream(run_id, after), media_type="text/event-stream"
    )


async def _event_stream(run_id: str, after: int):
    cursor = after
    while True:
        events = await runtime.events(run_id, after_sequence=cursor)
        for event in events:
            cursor = event.sequence_number
            yield (
                f"id: {cursor}\n"
                f"event: {event.event.get('type', 'message')}\n"
                f"data: {json.dumps(event.event)}\n\n"
            )

        state = await runtime.get(run_id)
        if state is None or state.status in {
            DurableExecutionStatus.COMPLETED,
            DurableExecutionStatus.FAILED,
        }:
            return
        await asyncio.sleep(0.25)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    )
