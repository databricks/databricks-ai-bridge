"""Hosted dispatcher for Claude Managed Agents on Databricks Sandbox, as a Databricks App.

Service mode (M1) always runs: the App's service principal long-polls the service environment ("S") and serves its
sessions. Routes (all behind the Apps proxy, so only users with CAN_USE on the App reach them):

  GET /            read-only status page
  GET /api/status  the same counters as JSON
  GET /api/health  200 while the poll loop runs, 503 otherwise (the Apps proxy answers /healthz itself)

With CMA_USER_MODE=1 (user mode, M2) the App also watches the user environment ("U") and serves the sign-in link
page in ``user_mode.py`` (``/connect``, ``/api/connect``). Sessions on env U run as the Databricks user who signs in.
See app/README.md.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from service import DispatcherService, from_environment
from user_mode import USER_MODE_ENV, add_routes, user_mode_from_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Request and header logging stays off: headers carry the forwarded user token.
for noisy in ("httpx", "httpx2", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("cma_dispatcher_app")

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(make_service=from_environment, make_user_mode=user_mode_from_environment) -> FastAPI:
    user_enabled = os.environ.get(USER_MODE_ENV) == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service: DispatcherService = make_service()
        app.state.service = service
        if user_enabled:
            # Misconfigured user mode fails the start-up here, not on the first user request.
            app.state.user = make_user_mode(service)
            try:
                await asyncio.to_thread(app.state.user.store.ensure_schema)
            except Exception as e:  # noqa: BLE001 - user routes answer 503 until the store works
                # StoreUnavailable messages are fixed strings (missing variable names, error types): safe.
                reason = str(e) if type(e).__name__ == "StoreUnavailable" else type(e).__name__
                log.warning("user-mode store not ready at start-up: %s", reason)
            await asyncio.to_thread(app.state.user.check_memory)
            app.state.user.start()
            log.info(
                "user mode: watching environment=%s; sign-in links %s; memory %s",
                app.state.user.config.environment_id,
                "on" if app.state.user.app_url else "off (no App URL)",
                app.state.user.memory_state,
            )
        service.start()
        log.info(
            "dispatcher started: environment=%s prefix=%s",
            service.launcher.environment_id,
            service.launcher.sandbox_id_prefix,
        )
        log.info("databricks user-agent: %s", service.launcher.workspace_client.config.user_agent)
        try:
            yield
        finally:
            if user_enabled:
                await app.state.user.stop()
            await service.stop()

    app = FastAPI(
        title="Claude Managed Agents dispatcher",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        # setdefault: a route may set its own policy (the /connect page's CSP allows its own script).
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    if user_enabled:
        add_routes(app)

    @app.get("/api/health")
    async def healthz(request: Request):
        healthy = request.app.state.service.healthy
        return JSONResponse({"healthy": healthy}, status_code=200 if healthy else 503)

    def _status(request: Request) -> dict:
        s = request.app.state.service.full_status()
        user = getattr(request.app.state, "user", None)
        if user is not None:
            s["user_memory"] = user.memory_state
        return s

    @app.get("/api/status")
    async def status(request: Request):
        return await asyncio.to_thread(_status, request)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        s = await asyncio.to_thread(_status, request)
        d = s.pop("dispatcher") or {}
        rows = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in {**s, **d}.items()
        )
        colour = "#1b7f3b" if s["healthy"] else "#b42318"
        return (
            "<!doctype html><meta charset=utf-8><title>CMA dispatcher</title>"
            "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}"
            "th,td{text-align:left;padding:4px 12px;border-bottom:1px solid #ddd}</style>"
            f"<h1>Claude Managed Agents dispatcher</h1><p style='color:{colour}'><b>"
            f"{'healthy' if s['healthy'] else 'unhealthy'}</b> · service mode</p><table>{rows}</table>"
        )

    return app


# Building the app reads no configuration; the dispatcher starts in the lifespan.
app = create_app()
