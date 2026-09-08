"""Request shaping and response semantics for known agent HTTP contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from databricks_mason.errors import AgentCliError


@dataclass(frozen=True)
class EndpointPreset:
    """Known defaults layered on top of the generic endpoint transport."""

    name: str
    path: str
    client_generated_id: bool


_PRESETS = {
    "mason": EndpointPreset(
        name="mason",
        path="/api/invocations",
        client_generated_id=False,
    ),
    "mason-durable": EndpointPreset(
        name="mason-durable",
        path="/api/invocations",
        client_generated_id=True,
    ),
}

PRESET_NAMES = tuple(_PRESETS)


def get_preset(name: str | None) -> EndpointPreset | None:
    """Return a configured preset, or ``None`` for generic HTTP mode."""
    if name is None:
        return None
    return _PRESETS[name]


def build_preset_body(
    preset: EndpointPreset,
    body: Any,
    *,
    message: str | None,
    stream: bool,
    background: bool,
    request_id: str | None = None,
    force_request_id: bool = False,
) -> dict[str, Any]:
    """Apply one preset's request envelope to a caller-provided JSON body."""
    if message is not None and body is not None:
        raise AgentCliError("--message cannot be combined with --json or --json-file.")
    if message is not None:
        payload: Any = {"input": [{"role": "user", "content": message}]}
    elif body is None:
        payload = {"input": []}
    else:
        payload = copy.deepcopy(body)
    if not isinstance(payload, Mapping):
        raise AgentCliError(
            f"The {preset.name} preset requires a JSON object request body.",
            hint="Wrap the agent input under an 'input' field.",
        )

    request = dict(payload)
    if stream:
        request["stream"] = True
    if background:
        request["background"] = True
    if preset.client_generated_id and (force_request_id or "id" not in request):
        request["id"] = request_id or str(uuid4())
    return request


def polling_path(preset: EndpointPreset, response: Mapping[str, Any]) -> str | None:
    """Return the preset-specific status path from an accepted response."""
    status_url = response.get("status_url")
    if isinstance(status_url, str) and status_url:
        return status_url
    invocation_id = response.get("id")
    if isinstance(invocation_id, str) and invocation_id:
        return f"{preset.path}/{invocation_id}"
    return None


def terminal_status(status: object) -> bool:
    """Whether a known Mason invocation status is terminal."""
    return str(status or "").lower() in {"completed", "failed", "error", "cancelled"}
