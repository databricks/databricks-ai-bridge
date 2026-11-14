"""Thin App: HTTP, Lakebase registry, and sandbox orchestration."""

import json
import os
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from anyio import CancelScope
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from databricks_agentkit.runtime.model_services import list_ai_gateway_model_services
from databricks_agentkit.runtime.store import RUNTIME_STORE_USERNAME_ENV
from databricks_agentkit.sandbox.bundle import bundle_contents, files_digest
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog
from databricks_agentkit.sandbox.control import ControlStore, scoped_session
from databricks_agentkit.sandbox.debug import terminal
from databricks_agentkit.sandbox.display import chat_snapshot
from databricks_agentkit.sandbox.environment import bind_environment, on_demand, saved_bundle
from databricks_agentkit.sandbox.environments import (
    EnvironmentExists,
    EnvironmentStore,
    validate_name,
)
from databricks_agentkit.sandbox.features import source_features
from databricks_agentkit.sandbox.jobs import RunManager
from databricks_agentkit.sandbox.lakebox import Lakebox
from databricks_agentkit.sandbox.orchestrator import Orchestrator, RevisionConflict
from databricks_agentkit.sandbox.preparation import SessionPreparation
from databricks_agentkit.sandbox.source_view import source_page
from databricks_agentkit.sandbox.sources import (
    DEFAULT_ENTRYPOINT,
    SourceLoader,
    source_path,
    validate_entrypoint,
)
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, Registry, SessionBusy

STATIC = Path(__file__).parent / "static"


class InvocationRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=512)
    invocation_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=512)
    input: dict = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app):
    workspace = WorkspaceClient()
    namespace = os.environ["SANDBOX_NAMESPACE"]
    registry = Registry(workspace, namespace)
    registry.initialize()
    app.state.orchestrator = Orchestrator(
        registry,
        Lakebox(
            workspace,
            Path(os.getenv("SANDBOX_STATE_DIR", ".state")),
            namespace,
            transport=os.getenv("SANDBOX_TRANSPORT", "ssh"),
            idle_timeout=int(os.getenv("SANDBOX_IDLE_TIMEOUT_SECONDS", "900")),
        ),
        Path(os.getenv("SANDBOX_AGENT_SOURCE", "agent_source")),
        namespace,
    )
    app.state.debug_transport = os.getenv(
        "SANDBOX_DEBUG_TRANSPORT", os.getenv("SANDBOX_TRANSPORT", "ssh")
    )
    if app.state.debug_transport not in {"ssh", "https"}:
        raise ValueError("SANDBOX_DEBUG_TRANSPORT must be ssh or https")
    app.state.debug_lakebox = (
        Lakebox(
            workspace,
            Path(os.getenv("SANDBOX_STATE_DIR", ".state")) / "debug",
            namespace,
            transport="ssh",
        )
        if app.state.debug_transport == "ssh"
        else app.state.orchestrator.lakebox
    )
    app.state.sources = SourceLoader(workspace, app.state.orchestrator.source)
    catalog = WorkspaceCatalog(
        workspace, os.getenv("SANDBOX_CATALOG_PATH", f"/Shared/agentbricks-runtime/{namespace}")
    )
    app.state.control = ControlStore(registry, catalog)
    app.state.control.initialize()
    app.state.environments = EnvironmentStore(app.state.control, app.state.sources)
    app.state.environments.initialize()
    app.state.prepared_environment = None
    if environment_file := os.getenv("AGENTBRICKS_ENVIRONMENT_FILE"):
        prepared = json.loads(Path(environment_file).read_text())
        bind_environment(app.state.orchestrator, prepared)
        app.state.prepared_environment = prepared
        bundle = saved_bundle(app.state.control, prepared)
        prepared.setdefault(
            "workbench", source_features(bundle_contents(bundle.data).get("agent.toml"))
        )
        if not on_demand(prepared):
            # Only legacy prepared environments probe their installation on startup.
            app.state.orchestrator.invoke(
                prepared["preparation_id"],
                "",
                {},
                bundle=bundle,
                entrypoint=prepared["entrypoint"],
                _prepare_only=True,
                _allow_install=False,
            )
    app.state.runs = RunManager(app.state.control, app.state.orchestrator)
    app.state.warm_pool = None
    pool_size = int(os.getenv("SANDBOX_WARM_POOL_SIZE", "0"))
    if pool_size and not on_demand(app.state.prepared_environment or {}):
        raise ValueError("Warm pools require an on-demand source deployment")
    if (
        os.getenv("SANDBOX_WARM_POOL_GENERATION")
        and on_demand(app.state.prepared_environment or {})
        and (pool_size or app.state.orchestrator.sandbox_scope == "session")
    ):
        from databricks_agentkit.sandbox.warm_pool import WarmPool

        app.state.warm_pool = WarmPool(
            app.state.orchestrator,
            prepared,
            bundle,
            size=pool_size,
            generation=int(os.environ["SANDBOX_WARM_POOL_GENERATION"]),
        )
        app.state.warm_pool.start()
    app.state.runs.start()
    app.state.preparations = SessionPreparation(app.state.control, app.state.orchestrator)
    app.state.preparations.start()
    try:
        yield
    finally:
        if app.state.warm_pool:
            app.state.warm_pool.stop()
        app.state.preparations.stop()
        app.state.runs.stop()
        app.state.control.close()


app = FastAPI(title="Agent Bricks sandbox launcher", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.add_api_websocket_route("/api/environments/{environment}/ssh", terminal)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.exception_handler(DatabricksError)
def platform_error(request, exc):
    return JSONResponse(
        status_code=403
        if isinstance(exc, PermissionDenied)
        else 404
        if isinstance(exc, NotFound)
        else 502,
        content={"detail": f"{type(exc).__name__}: {exc}"[:4000]},
    )


@app.exception_handler(SessionBusy)
def session_busy(request, exc):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(KeyError)
def missing_record(request, exc):
    return JSONResponse(status_code=404, content={"detail": exc.args[0]})


@app.exception_handler(ValueError)
def invalid_request(request, exc):
    return JSONResponse(
        status_code=409
        if isinstance(exc, (EnvironmentExists, ExecutionIdentityConflict, RevisionConflict))
        else 400,
        content={"detail": str(exc)[:4000]},
    )


class AgentRegistration(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_kind: Literal["workspace", "volume", "bundled"]
    source_path: str = Field(default="", max_length=2048)
    entrypoint: str = Field(default=DEFAULT_ENTRYPOINT, max_length=256)


class AgentInvocation(InvocationRequest):
    agent_id: str = Field(min_length=1, max_length=64)
    retry: bool = False


class GitSource(BaseModel):
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    requested_revision: str = Field(min_length=1, max_length=1024)
    repository_url: str | None = Field(default=None, max_length=2048)
    subdirectory: str = Field(default="", max_length=2048)
    files_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("repository_url")
    @classmethod
    def public_repository_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "Repository URL must be HTTPS without credentials, query, or fragment"
                )
        return value


class EnvironmentRegistration(AgentRegistration):
    source_kind: Literal["workspace", "volume"] = "workspace"
    git_source: GitSource | None = None


class EnvironmentInvocation(InvocationRequest):
    retry: bool = False


@app.post("/api/environments", status_code=201)
def create_environment(request: EnvironmentRegistration):
    if getattr(app.state, "prepared_environment", None):
        raise HTTPException(
            409,
            "This deployment is bound to an environment; publish and deploy another environment to change source",
        )
    name = validate_name(request.name)
    entrypoint = validate_entrypoint(request.entrypoint)
    path = source_path(request.source_kind, request.source_path)
    # Verify App-principal access and required project files before registration.
    bundle = app.state.sources.load(request.source_kind, path)
    git_source = request.git_source.model_dump() if request.git_source else None
    if git_source and files_digest(bundle_contents(bundle.data)) != git_source["files_digest"]:
        raise ValueError(
            "Uploaded source differs from the selected Git snapshot; create again in a new folder"
        )
    return app.state.environments.create(
        name, request.source_kind, path, entrypoint, bundle=bundle, git_source=git_source
    )


@app.get("/api/environments")
def list_environments():
    return app.state.environments.list()


@app.get("/api/environments/{name}")
def get_environment(name: str):
    return app.state.environments.get(name)


@app.post("/api/environments/{name}/runs", status_code=202)
def invoke_environment(name: str, request: EnvironmentInvocation):
    if preparations := getattr(app.state, "preparations", None):
        environment = app.state.environments.get(name)
        preparations.assert_invocable(environment["default_agent_id"], request.session_id)
    row = app.state.environments.invoke(
        name, request.session_id, request.invocation_id, request.input, retry=request.retry
    )
    if row["status"] == "queued":
        app.state.runs.submit(row["id"])
    return chat_snapshot(row)


@app.get("/api/config")
def configuration():
    return {
        "workspace": app.state.orchestrator.lakebox.workspace.config.host,
        "service_principal": os.getenv(RUNTIME_STORE_USERNAME_ENV) or os.getenv("PGUSER", ""),
        "runtime_store": (
            (getattr(app.state, "prepared_environment", None) or {}).get("runtime_store") or {}
        ).get("name"),
        "example_workspace_path": os.getenv("SANDBOX_EXAMPLE_WORKSPACE_PATH", ""),
        "default_entrypoint": (getattr(app.state, "prepared_environment", None) or {}).get(
            "entrypoint", DEFAULT_ENTRYPOINT
        ),
        "default_environment": (getattr(app.state, "prepared_environment", None) or {}).get("name"),
        "default_agent_id": (getattr(app.state, "prepared_environment", None) or {}).get(
            "agent_id"
        ),
        "sandbox_idle_timeout_seconds": app.state.orchestrator.lakebox.idle_timeout,
        "sandbox_warm_pool_size": int(os.getenv("SANDBOX_WARM_POOL_SIZE", "0")),
        "sandbox_scope": app.state.orchestrator.sandbox_scope,
        "environment_mode": (
            "on-demand" if on_demand(app.state.prepared_environment) else "prepared"
        )
        if getattr(app.state, "prepared_environment", None)
        else None,
        "capabilities": [
            "git-revisions",
            "session-pinning",
            "source-snapshots",
            "text-streaming",
            "session-sandboxes",
            "session-ssh",
            "session-preparation",
        ],
        "workbench": (getattr(app.state, "prepared_environment", None) or {}).get("workbench"),
    }


@app.get("/api/warm-pool")
def warm_pool_status():
    pool = getattr(app.state, "warm_pool", None)
    return (
        pool.store.status()
        if pool
        else {"maximum": 0, "ready": 0, "preparing": 0, "retiring": 0, "error": None}
    )


@app.get("/api/agents")
def list_agents():
    return app.state.control.agents()


@app.get("/api/models")
def list_models():
    # Match the template UI's discovery, using the runtime's execution identity.
    # This read does not import the agent or create a sandbox.
    return {"available": list_ai_gateway_model_services(app.state.orchestrator.lakebox.workspace)}


@app.get("/api/agents/{agent_id}/source", response_class=HTMLResponse)
def view_source(agent_id: str, path: str | None = None):
    agent = app.state.control.agent(agent_id, with_bundle=True)
    return HTMLResponse(source_page(agent, path), headers={"Cache-Control": "no-store"})


@app.post("/api/agents", status_code=201)
def register_agent(request: AgentRegistration):
    if getattr(app.state, "prepared_environment", None):
        raise HTTPException(409, "Publish a new environment and deploy it to change source")
    entrypoint = validate_entrypoint(request.entrypoint)
    path = (
        source_path(request.source_kind, request.source_path)
        if request.source_kind != "bundled"
        else ""
    )
    bundle = app.state.sources.load(request.source_kind, path)
    return app.state.control.create_agent(
        request.name, request.source_kind, path, entrypoint, bundle
    )


@app.get("/api/agents/{agent_id}/sessions")
def list_sessions(agent_id: str):
    app.state.control.agent(agent_id)
    sessions = {row["session_id"]: row for row in app.state.control.sessions(agent_id)}
    for row in app.state.preparations.rows(agent_id=agent_id):
        prepared = row["preparation"]
        sessions.setdefault(prepared["session_id"], prepared)
    return list(sessions.values())


class CreateSessionRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=512)
    retry: bool = False


@app.post("/api/agents/{agent_id}/sessions", status_code=202)
def create_session(agent_id: str, request: CreateSessionRequest):
    return app.state.preparations.create(agent_id, request.session_id, retry=request.retry)


@app.get("/api/agents/{agent_id}/sessions/{session_id}")
def preparation_status(agent_id: str, session_id: str):
    app.state.control.agent(agent_id)
    return app.state.preparations.get(agent_id, session_id)


@app.get("/api/agents/{agent_id}/sandbox")
def session_sandbox(agent_id: str, session_id: str):
    """Read the current session mapping and compute status without starting it."""
    agent = app.state.control.agent(agent_id)
    orchestrator = app.state.orchestrator
    key = orchestrator.execution_session_id(scoped_session(agent_id, session_id))
    runtime = orchestrator.sandboxes.status(key)
    owner, row = runtime["execution_principal"], runtime["sandbox"]
    sandbox_id, status = runtime["sandbox_id"], runtime["status"]
    return {
        "preparation": ((row or {}).get("bootstrap_metrics") or {}).get("preparation"),
        "session_id": session_id,
        "agent_id": agent_id,
        "sandbox_scope": orchestrator.sandbox_scope,
        "sandbox_id": sandbox_id,
        "status": status,
        "execution_principal": owner,
        "warm_pool": ((row or {}).get("bootstrap_metrics") or {}).get("warm_pool"),
        "source_hash": row["source_hash"] if row else agent["source_hash"],
        "initial_bootstrap_timings_ms": ((row or {}).get("bootstrap_metrics") or {}).get(
            "timings_ms"
        ),
    }


@app.get("/api/runs")
def run_history(agent_id: str, session_id: str):
    return [chat_snapshot(row) for row in app.state.control.history(agent_id, session_id)]


@app.post("/api/runs", status_code=202)
def launch_agent(request: AgentInvocation):
    app.state.control.agent(request.agent_id)
    if preparations := getattr(app.state, "preparations", None):
        preparations.assert_invocable(request.agent_id, request.session_id)
    row = app.state.control.create_run(
        request.agent_id,
        request.session_id,
        request.invocation_id,
        request.input,
        retry=request.retry,
    )
    if row["status"] == "queued":
        app.state.runs.submit(row["id"])
    return chat_snapshot(row)


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    return chat_snapshot(app.state.control.run(run_id))


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str):
    # Resolve errors before sending HTTP 200. Subscription never launches work.
    app.state.control.run(run_id)

    async def events():
        import json

        previous = None
        watch = app.state.control.watch(run_id)

        def next_snapshot() -> dict | None:
            return next(watch, None)

        try:
            while (row := await run_in_threadpool(next_snapshot)) is not None:
                payload = json.dumps(jsonable_encoder(chat_snapshot(row)), ensure_ascii=True)
                if payload != previous:
                    # Full snapshots are idempotent on reconnect, even across retries.
                    yield f"event: run\ndata: {payload}\n\n"
                    previous = payload
                else:
                    yield ": keepalive\n\n"
        finally:
            close = getattr(watch, "close", None)
            if close:
                # Client disconnect cancels the response task; still release its
                # tail reader after the in-flight snapshot read finishes.
                with CancelScope(shield=True):
                    await run_in_threadpool(close)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=512)
    agent_id: str | None = Field(default=None, max_length=64)


@app.post("/sessions/stop")
def stop_session(request: SessionRequest):
    """Stop only a sandbox owned by this prototype's registry; preserve its disk."""
    orchestrator = app.state.orchestrator
    key = (
        scoped_session(request.agent_id, request.session_id)
        if request.agent_id
        else request.session_id
    )
    key = orchestrator.execution_session_id(key)
    try:
        preparations = getattr(app.state, "preparations", None)
        with (
            preparations.stopping_session(request.agent_id, request.session_id)
            if preparations and request.agent_id
            else nullcontext()
        ):
            return orchestrator.sandboxes.stop(key)
    except SessionBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "execution": "lakebox", "registry": "lakebase"}


@app.post("/invoke")
def invoke(request: InvocationRequest):
    try:
        return app.state.orchestrator.invoke(
            request.session_id, request.invocation_id, request.input
        )
    except (SessionBusy, RevisionConflict) as exc:
        raise HTTPException(409, str(exc)) from exc
