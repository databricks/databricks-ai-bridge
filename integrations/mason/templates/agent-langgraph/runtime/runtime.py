"""The agent's HTTP surface: a hand-written, SDK-agnostic FastAPI app.

``build_app`` wires the endpoints to two handlers with a generic contract:

    invoke_handler(request: dict) -> dict
    stream_handler(request: dict) -> AsyncGenerator[dict]

Nothing here is SDK-specific — the agent lives entirely behind those handlers (``agent/agent.py``).
The Databricks Apps ``__Host-databricks-app-router`` cookie is both the replica-affinity key and the
application session id. Clients never send ``session_id`` in the JSON body. Local development uses
an HTTP-only fallback cookie because the Apps router is not present.

TODO: Prefer ``X-Routing-Key`` for API clients once Databricks Apps supports it; until then, use the
documented router cookie for sticky routing.

Endpoints: ``POST /invocations`` (``stream: true`` → SSE ending with ``data: [DONE]``;
``background: true`` → an ``invocation_id`` to poll), ``GET /invocations/{invocation_id}`` to poll a
background run, ``POST /api/session/new`` to rotate the routing session, and ``GET /health``. The
invocation and health routes also have ``/api`` aliases because Databricks Apps accepts programmatic
Bearer-token authentication only on paths under ``/api/``. Each request is wrapped in an MLflow span
for tracing.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable

import mlflow
from databricks_mason.runtime.background import BackgroundRuns
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from uuid_utils import uuid7

# Request keys that control transport; stripped before the request reaches the handler.
_REQUEST_STREAM_PARAM_KEY = "stream"
_REQUEST_BACKGROUND_PARAM_KEY = "background"
_REQUEST_SESSION_ID_HEADER_KEY = (
    "X-Routing-Key"  # carries the session id (generic for Apps + Agents)
)
_TRACE_NAME_TAG = "mlflow.traceName"
_ROUTING_COOKIE = "__Host-databricks-app-router"
_LOCAL_SESSION_COOKIE = "mason-local-session"
# Forwarded-identity headers for the signed-in user, in priority order — the deployment platform
# sets these; absent locally. Their value becomes the request's actor (memory / session partition).
_USER_HEADERS = ("X-Forwarded-Email", "X-Forwarded-User")

# TODO: Replace the Apps routing cookie with X-Routing-Key when Databricks Apps supports it.

InvokeHandler = Callable[[dict], Awaitable[dict]]
StreamHandler = Callable[[dict], AsyncGenerator[dict, None]]


def _sse(data: dict | str) -> str:
    return f"data: {json.dumps(data) if isinstance(data, dict) else data}\n\n"


def _set_trace_name(name: str) -> None:
    if mlflow.get_current_active_span() is not None:
        mlflow.update_current_trace(tags={_TRACE_NAME_TAG: name})


def rotate_session_cookie(request: Request, response: Response, session_id: str) -> None:
    if request.cookies.get(_ROUTING_COOKIE):
        response.set_cookie(
            _ROUTING_COOKIE,
            session_id,
            secure=True,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.delete_cookie(_LOCAL_SESSION_COOKIE, path="/")
    elif request.cookies.get(_LOCAL_SESSION_COOKIE):
        response.set_cookie(
            _LOCAL_SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="lax",
            path="/",
        )


def build_app(invoke_handler: InvokeHandler, stream_handler: StreamHandler) -> FastAPI:
    """Build the FastAPI app wiring the endpoints to the agent's invoke/stream handlers."""
    app = FastAPI(title="Agent Server")
    runs = BackgroundRuns()

    @app.middleware("http")
    async def bind_session(request: Request, call_next):
        routing_session = request.cookies.get(_ROUTING_COOKIE)
        local_session = request.cookies.get(_LOCAL_SESSION_COOKIE)
        request.state.session_id = routing_session or local_session or str(uuid7())
        response = await call_next(request)
        if not routing_session and not local_session:
            response.set_cookie(
                _LOCAL_SESSION_COOKIE,
                request.state.session_id,
                httponly=True,
                samesite="lax",
            )
        return response

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
        data.pop("session_id", None)
        data["session_id"] = request.state.session_id
        # The signed-in user (from the platform's forwarded-identity headers) becomes the request's
        # actor, so the agent partitions memory + session data per user. Absent (e.g. local dev) → the
        # handler defaults it. Clients can't set it: any body value is dropped in favor of the header.
        data.pop("actor", None)
        actor = next(
            (v for h in _USER_HEADERS if (v := request.headers.get(h))),
            None,
        )
        if actor:
            data["actor"] = actor

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

    @app.post("/api/session/new")
    async def new_session(request: Request) -> JSONResponse:
        previous_session_id = request.state.session_id
        session_id = str(uuid7())
        request.state.session_id = session_id
        response = JSONResponse(
            {
                "session_id": session_id,
                "previous_session_id": previous_session_id,
            }
        )
        rotate_session_cookie(request, response, session_id)
        return response

    return app
