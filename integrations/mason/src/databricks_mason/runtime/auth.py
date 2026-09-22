"""Transient authentication for request-bound, user-authorized tool calls."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, SupportsIndex

from databricks_mason.runtime.store import RUNTIME_STORE_LOCAL_ENV

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


class AuthError(RuntimeError):
    """A credential-free error suitable for returning to the caller."""

    def __init__(
        self, code: str, message: str, status_code: int = 401, integration_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.integration_id = integration_id

    def payload(self) -> dict[str, str]:
        result = {"code": self.code, "message": str(self)}
        if self.integration_id is not None:
            result["integration_id"] = self.integration_id
        return result


@dataclass(frozen=True)
class InvocationAuthPolicy:
    """Whether an invocation needs credentials scoped to its active execution attempt."""

    user_tools: tuple[str, ...] = ()

    @property
    def requires_user(self) -> bool:
        return bool(self.user_tools)


class RequestAuthContext:
    """Private request credentials, never part of a persisted invocation payload.

    Deployed Apps must only be reachable through the trusted Apps ingress, which supplies the
    forwarded identity headers. Local development deliberately ignores those headers.
    """

    __slots__ = (
        "_token",
        "_principal",
        "_host",
        "_app",
        "_workspace",
        "_local",
        "_clients",
        "_closed",
    )

    def __init__(self, *, token: str | None, principal: str, local: bool) -> None:
        self._token = token
        self._principal = principal
        self._host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        self._app = os.getenv("DATABRICKS_APP_NAME", "local")
        self._workspace = os.getenv("DATABRICKS_WORKSPACE_ID", "")
        self._local = local
        self._clients: dict[str, WorkspaceClient] = {}
        self._closed = False

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> RequestAuthContext:
        local_runtime = os.getenv(RUNTIME_STORE_LOCAL_ENV, "").lower() == "true"
        if local_runtime or not os.getenv("DATABRICKS_APP_NAME"):
            return cls(token=None, principal="local-developer", local=True)
        normalized = {name.lower(): value for name, value in headers.items()}
        token = normalized.get("x-forwarded-access-token", "").strip()
        principal = normalized.get("x-forwarded-user", "").strip()
        if not token:
            raise AuthError("MCP_USER_AUTH_REQUIRED", "Missing request-user access token")
        if not principal:
            raise AuthError("MCP_USER_IDENTITY_MISSING", "Missing trusted request-user identity")
        return cls(token=token, principal=principal, local=False)

    def __repr__(self) -> str:
        return f"RequestAuthContext(closed={self._closed})"

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("RequestAuthContext cannot be serialized")

    def namespace(self, kind: str, value: str) -> str:
        material = json.dumps(
            [self._host, self._workspace, self._app, self._principal, kind, value],
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def client_for(self, mode: str) -> WorkspaceClient:
        if self._closed:
            raise AuthError(
                "MCP_USER_AUTH_EXPIRED", "Request-user authentication is no longer active"
            )
        if mode not in ("user", "app"):
            raise ValueError("auth must be 'user' or 'app'")
        if mode not in self._clients:
            from databricks_mason.runtime.workspace import workspace_client, workspace_headers

            if mode == "app" or self._local:
                self._clients[mode] = workspace_client()
            else:
                from databricks.sdk import WorkspaceClient

                if not self._host:
                    raise AuthError(
                        "MCP_USER_AUTH_CONFIGURATION", "Workspace host is not configured", 500
                    )
                self._clients[mode] = WorkspaceClient(
                    host=self._host,
                    token=self._token,
                    auth_type="pat",
                    custom_headers=workspace_headers(),
                )
        return self._clients[mode]

    def close(self) -> None:
        """Drop credentials and prohibit further client resolution."""
        self._closed = True
        self._token = None
        self._clients.clear()
