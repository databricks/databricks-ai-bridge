"""The agent's HTTP surface: a hand-written, SDK-agnostic FastAPI app.

``build_app`` wires the endpoints to two handlers with a generic contract:

    invoke_handler(request: dict) -> dict
    stream_handler(request: dict) -> AsyncGenerator[dict]

Nothing here is SDK-specific — the agent lives entirely behind those handlers (``agent/agent.py``).
The session id (used for multi-turn and to resume a paused run) travels in the JSON body as
``session_id``. Databricks Apps replica affinity is handled separately by the
``__Host-databricks-app-router`` cookie.

Endpoints: ``POST /invocations`` (``stream: true`` → SSE ending with ``data: [DONE]``;
``background: true`` → an ``invocation_id`` to poll), ``GET /invocations/{invocation_id}`` to poll a
background run, and ``GET /health``. Each route also has an ``/api`` alias because Databricks Apps
accepts programmatic Bearer-token authentication only on paths under ``/api/``. Each request is
wrapped in an MLflow span for tracing.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable

import mlflow
from agent.mason.background import BackgroundRuns
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Request keys that control transport; stripped before the request reaches the handler.
_REQUEST_STREAM_PARAM_KEY = "stream"
_REQUEST_BACKGROUND_PARAM_KEY = "background"
_REQUEST_SESSION_ID_HEADER_KEY = (
    "X-Routing-Key"  # carries the session id (generic for Apps + Agents)
)
_TRACE_NAME_TAG = "mlflow.traceName"

InvokeHandler = Callable[[dict], Awaitable[dict]]
StreamHandler = Callable[[dict], AsyncGenerator[dict, None]]


def _sse(data: dict | str) -> str:
    return f"data: {json.dumps(data) if isinstance(data, dict) else data}\n\n"


def _set_trace_name(name: str) -> None:
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(tags={_TRACE_NAME_TAG: name})


def build_app(invoke_handler: InvokeHandler, stream_handler: StreamHandler) -> FastAPI:
    """Build the FastAPI app wiring the endpoints to the agent's invoke/stream handlers."""
    app = FastAPI(title="Agent Server")
    runs = BackgroundRuns()

    async def _invoke(request: dict) -> dict:
        with mlflow.start_span(name="invoke_handler") as span:
            _set_trace_name("invoke_handler")
            span.set_inputs(request)
            result = await invoke_handler(request)
            span.set_outputs(result)
            return result

    async def _stream(request: dict) -> AsyncGenerator[str, None]:
        with mlflow.start_span(name="stream_handler") as span:
            _set_trace_name("stream_handler")
            span.set_inputs(request)
            chunks: list[dict] = []
            try:
                async for chunk in stream_handler(request):
                    chunks.append(chunk)
                    yield _sse(chunk)
                span.set_outputs(chunks)
            except Exception as e:  # surface the error in-band, then close the stream
                yield _sse({"error": str(e)})
            yield _sse("[DONE]")

    async def _run_background(invocation_id: str, request: dict) -> None:
        try:
            runs.complete(invocation_id, await _invoke(request))
        except Exception as e:
            runs.fail(invocation_id, str(e))

    @app.post("/api/invocations")
    @app.post("/invocations")
    async def invoke(request: Request):
        data = await request.json()
        is_stream = bool(data.pop(_REQUEST_STREAM_PARAM_KEY, False))
        is_background = bool(data.pop(_REQUEST_BACKGROUND_PARAM_KEY, False))

        if is_background:
            invocation_id = runs.start()
            # Fire-and-forget; the task updates `runs` when it finishes. Non-durable (in-memory).
            asyncio.create_task(_run_background(invocation_id, data))
            return JSONResponse({"id": invocation_id, "status": "in_progress"})
        if is_stream:
            return StreamingResponse(_stream(data), media_type="text/event-stream")
        return JSONResponse(await _invoke(data))

    @app.get("/api/invocations/{invocation_id}")
    @app.get("/invocations/{invocation_id}")
    async def retrieve(invocation_id: str):
        run = runs.get(invocation_id)
        if run is None:
            return JSONResponse({"error": "unknown invocation id"}, status_code=404)
        if run["status"] == "completed":
            return JSONResponse({"id": invocation_id, "status": "completed", **run["output"]})
        return JSONResponse({"id": invocation_id, "status": run["status"], "error": run["error"]})

    @app.get("/api/health")
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
