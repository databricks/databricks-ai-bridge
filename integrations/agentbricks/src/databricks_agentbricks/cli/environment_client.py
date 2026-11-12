"""HTTP client for the experimental sandbox launcher, separate from managed agent APIs."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from databricks_agentbricks.errors import AgentCliError, wrap_api_error
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.invoke.transport import EndpointRequest, HttpSession
from databricks_agentkit._api_client import _workspace_client


class LauncherClient:
    def __init__(self, profile, *, app=None, url=None, auth=None):
        if bool(app) == bool(url):
            raise AgentCliError("Provide exactly one of --app or --url for the sandbox launcher.")
        self.profile = profile
        self._workspace = None
        self.app = app
        if app:
            try:
                url = self.workspace.apps.get(app).url
            except Exception as exc:  # noqa: BLE001 - SDK boundary
                raise wrap_api_error(exc) from exc
        parsed = urlsplit(url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AgentCliError("The sandbox launcher must have an HTTP or HTTPS URL.")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise AgentCliError("Launcher URLs cannot contain credentials, queries, or fragments.")
        self.url = (url or "").rstrip("/")
        self.authenticate = bool(app) if auth is None else auth
        self.session = HttpSession()

    @property
    def workspace(self):
        if self._workspace is None:
            try:
                self._workspace = _workspace_client(self.profile)
            except Exception as exc:  # noqa: BLE001 - SDK authentication boundary
                raise wrap_api_error(exc) from exc
        return self._workspace

    def request(self, method, path, body=None, *, missing_ok=False, routing_key=None):
        headers = {"Content-Type": "application/json"}
        if self.authenticate:
            headers["Authorization"] = WorkspaceOAuthAuthenticator(
                self.profile
            ).authorization_header()
        if routing_key:
            headers["X-Routing-Key"] = routing_key
        response = self.session.send(
            EndpointRequest(
                url=self.url + path,
                method=method,
                headers=headers,
                body=body,
                body_set=body is not None,
                timeout=60,
            )
        )
        if missing_ok and response.status_code == 404:
            return None
        if not 200 <= response.status_code < 300:
            raise AgentCliError(
                f"Sandbox launcher returned HTTP {response.status_code}.",
                hint=json.dumps(response.body, default=str)[:2000],
            )
        if not isinstance(response.body, (dict, list)):
            raise AgentCliError(
                "The endpoint did not return sandbox launcher JSON.",
                hint="Select the sandbox launcher App. For an authenticated --url, add --auth.",
            )
        return response.body
