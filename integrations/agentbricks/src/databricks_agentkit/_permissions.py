"""Shared store permission values and request validation."""

from __future__ import annotations

from dataclasses import dataclass

MAX_PERMISSION_PAGE_SIZE = 1000


@dataclass(frozen=True)
class Principal:
    """A user email or service principal application (client) ID."""

    principal_type: str
    name: str


@dataclass(frozen=True)
class StorePermission:
    """A caller's effective permission or a direct grant returned by List."""

    principal: Principal
    permission: str

    @classmethod
    def _from_response(cls, response: dict) -> StorePermission:
        principal = response["principal"]
        return cls(
            principal=Principal(principal_type=principal["principal_type"], name=principal["name"]),
            permission=response.get("permission", "PERMISSION_LEVEL_UNSPECIFIED"),
        )


def principal_body(principal_id: str, principal_type: str) -> dict[str, str]:
    if principal_type not in {"USER", "SERVICE_PRINCIPAL"}:
        raise ValueError("principal_type must be USER or SERVICE_PRINCIPAL")
    if not isinstance(principal_id, str) or not principal_id.strip():
        raise ValueError("principal_id must be a non-empty user email or application ID")
    return {"principal_type": principal_type, "name": principal_id}


def validate_permission(permission: str) -> None:
    if permission not in {"READ", "WRITE"}:
        raise ValueError("permission must be READ or WRITE")
