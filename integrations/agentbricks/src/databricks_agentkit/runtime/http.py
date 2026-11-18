"""Public invocation HTTP models and serialization, independent of server construction."""

import copy
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from pydantic import JsonValue as PydanticJsonValue

from databricks_agentkit.runtime.types import Invocation, InvocationStatus, JsonObject

_API_ROOT = "/api/invocations"
_UNKNOWN_FAILURE: JsonObject = {"code": "AGENT_ERROR", "message": "agent invocation failed"}


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    session_id: str | None = Field(default=None, min_length=1)
    input: PydanticJsonValue = Field(default_factory=list)
    background: bool = False
    stream: bool = False


def accepted_payload(
    state: Invocation, *, stream: bool, invocation_id: str | None = None
) -> JsonObject:
    invocation_id = invocation_id or state.invocation_id
    payload: JsonObject = {
        "id": invocation_id,
        "status": state.status.value.lower(),
        "status_url": f"{_API_ROOT}/{invocation_id}",
    }
    if stream:
        payload["events_url"] = f"{_API_ROOT}/{invocation_id}/events"
    return payload


def state_payload(state: Invocation, *, invocation_id: str | None = None) -> JsonObject:
    payload: JsonObject = {
        "id": invocation_id or state.invocation_id,
        "status": state.status.value.lower(),
    }
    if state.status == InvocationStatus.COMPLETED:
        payload["output"] = copy.deepcopy(state.response)
    elif state.status == InvocationStatus.FAILED:
        payload["error"] = failure_response(state)[0]
    return payload


def failure_response(state: Invocation | None) -> tuple[JsonObject, int, bool]:
    """Return a failed invocation's error, HTTP status, and whether the agent raised it.

    Auth failures (stored without ``source``) keep their original ``{"error": ...}`` body.
    """
    fallback: tuple[JsonObject, int, bool] = (dict(_UNKNOWN_FAILURE), 500, True)
    if state is None or state.status != InvocationStatus.FAILED:
        return fallback
    if not isinstance(state.response, dict):
        return fallback
    error = state.response.get("error")
    status_code = state.response.get("status_code")
    if not isinstance(error, dict) or not isinstance(status_code, int):
        return fallback
    if not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
        return fallback
    if status_code < 400 or status_code > 599:
        return fallback
    return copy.deepcopy(error), status_code, state.response.get("source") == "agent"
