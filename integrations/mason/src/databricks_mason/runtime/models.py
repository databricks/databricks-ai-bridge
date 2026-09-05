"""Discover the Unity Catalog AI Gateway chat models available in a workspace.

The AI Gateway exposes Databricks-managed models as Unity Catalog *model services* in the
``system.ai`` schema (e.g. ``system.ai.claude-sonnet-4-5``), queryable through an OpenAI-compatible
endpoint at ``<host>/ai-gateway/mlflow/v1``. This lists the chat-capable ones so the demo UI's model
picker can offer them; the agent then calls the chosen one with ``use_ai_gateway=True``.

Kept framework-neutral (no agent SDK, no ``mlflow``) so it sits with the other neutral runtime
helpers and can be listed from either template's ``runtime/ui.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

# The Unity Catalog REST route that lists model services under a schema.
_MODEL_SERVICES_PATH = "/api/2.1/unity-catalog/model-services"
# The AI Gateway's Databricks-managed models live in this UC schema.
_SYSTEM_AI_SCHEMA = "system.ai"
_PAGE_SIZE = 200
# Resource-name prefix the API returns, e.g. "model-services/system.ai.claude-sonnet-4-5".
_NAME_PREFIX = "model-services/"
# The list call can intermittently fail; retry a few times so a transient blip doesn't blank the
# picker. A non-transient error (e.g. no permission) simply fails every attempt, then surfaces.
_LIST_ATTEMPTS = 3
_RETRY_DELAY_S = 0.5


def _is_chat_capable(supported_api_types: Any) -> bool:
    """True if a model service speaks a chat/completions API (i.e. not embeddings-only).

    ``supported_api_types`` looks like ``["openai/v1/chat/completions"]``. Be lenient when it's
    absent (offer the model), but drop services that only advertise embeddings.
    """
    types = [str(t).lower() for t in (supported_api_types or [])]
    if not types:
        return True
    return any(("chat" in t or "completions" in t or "responses" in t) for t in types)


def _list_page(client: WorkspaceClient, query: dict[str, Any]) -> Any:
    """Fetch one page of the model-services list, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(_LIST_ATTEMPTS):
        try:
            return client.api_client.do("GET", _MODEL_SERVICES_PATH, query=query)
        except Exception as exc:  # noqa: BLE001 - retry any list failure, then surface the last one
            last_error = exc
            if attempt < _LIST_ATTEMPTS - 1:
                time.sleep(_RETRY_DELAY_S)
    raise last_error if last_error else RuntimeError("model-services list returned nothing")


def list_ai_gateway_models(client: WorkspaceClient) -> list[str]:
    """Return the fully-qualified names of chat-capable ``system.ai`` AI Gateway models, sorted.

    Each name is a Unity Catalog model-service path like ``system.ai.claude-sonnet-4-5`` — exactly
    the string the OpenAI-compatible gateway (`use_ai_gateway=True`) expects as its ``model``. Pages
    through the model-services list API, retrying transient list failures per page.

    Raises whatever the underlying request raises after retries (e.g. a permission error); callers
    that want a graceful fallback to just the agent's default model should catch it.
    """
    names: list[str] = []
    page_token: str | None = None
    while True:
        query: dict[str, Any] = {"parent": f"schemas/{_SYSTEM_AI_SCHEMA}", "page_size": _PAGE_SIZE}
        if page_token:
            query["page_token"] = page_token
        raw = _list_page(client, query)
        if not isinstance(raw, dict):
            break
        response = cast("dict[str, Any]", raw)
        for service in response.get("model_services") or []:
            if not isinstance(service, dict):
                continue
            name = str(service.get("name") or "")
            if name.startswith(_NAME_PREFIX):
                name = name[len(_NAME_PREFIX) :]
            if name and _is_chat_capable(service.get("supported_api_types")):
                names.append(name)
        page_token = response.get("next_page_token") or None
        if not page_token:
            break
    return sorted(names)
