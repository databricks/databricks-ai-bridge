"""Hosted dispatcher for Claude Managed Agents on Databricks Sandbox, as a Databricks App (M1, service mode).

Routes (all behind the Apps proxy, so only users with CAN_USE on the App reach them):

  GET /            read-only status page
  GET /api/status  the same counters as JSON
  GET /api/health  200 while the poll loop runs, 503 otherwise (the Apps proxy answers /healthz itself)

There are no write routes in M1: sessions are created in the Claude Console or with the Anthropic API on the
service-mode environment, and this App serves them. See docs/design/hosted-dispatcher.md.
"""

from __future__ import annotations

import asyncio
import html
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from service import DispatcherService, from_environment

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


def create_app(make_service=from_environment) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service: DispatcherService = make_service()
        app.state.service = service
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
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/api/health")
    async def healthz(request: Request):
        healthy = request.app.state.service.healthy
        return JSONResponse({"healthy": healthy}, status_code=200 if healthy else 503)

    @app.get("/api/status")
    async def status(request: Request):
        return await asyncio.to_thread(request.app.state.service.full_status)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        s = await asyncio.to_thread(request.app.state.service.full_status)
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
