"""Discover the Unity Catalog AI Gateway chat model services available in a workspace.

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
# The list API rejects page_size above 100 (InvalidParameterValue), so use its maximum.
_PAGE_SIZE = 100
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


def list_ai_gateway_model_services(client: WorkspaceClient) -> list[str]:
    """Return the names of chat-capable ``system.ai`` AI Gateway model services, sorted.

    Each name is a Unity Catalog model-service path like ``system.ai.claude-sonnet-4-5`` — exactly
    the string the OpenAI-compatible gateway (``use_ai_gateway=True``) expects as its ``model``.
    Pages through the model-services list API, retrying transient list failures per page.

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


# --- Managing a user-owned model service ------------------------------------------------------
#
# An agent can call a model service it owns (``catalog.schema.name``) instead of a ``system.ai.*``
# model directly, so the model behind it can be repointed without touching the agent. These helpers
# are the one place the request shapes live; the CLI's API client and the model-upgrade optimizer
# both build on them. Shapes follow the UC model-services API: a single pay-per-token destination
# nested under ``config.routing.destinations``.
#
# Two names per model: agents call the *model service* ``system.ai.claude-sonnet-4-5``, but a
# destination references the UC *registered model* behind it, ``models/system.ai.databricks-claude-
# sonnet-4-5``. Callers here always use the model-service name; ``foundation_model`` resolves the
# registered model (by reading the system service's own destination) and ``destination_model``
# maps it back.

_PERMISSIONS_PATH = "/api/2.1/unity-catalog/permissions"
_PPT_DESTINATION = "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL"
_SYSTEM_AI_PREFIX = f"{_SYSTEM_AI_SCHEMA}."
# Registered foundation models behind system.ai chat services carry this leaf prefix.
_FOUNDATION_PREFIX = "databricks-"
_foundation_cache: dict[str, str] = {}
# Suffix of the temporary clone the model-upgrade optimizer evaluates candidates against. An
# underscore, not a hyphen: the clone's name is a UC identifier.
EXPERIMENT_SUFFIX = "_exp"


def system_ai_name(model: str) -> str:
    """``claude-sonnet-4-5`` / ``system.ai.claude-sonnet-4-5`` → ``system.ai.claude-sonnet-4-5``."""
    return model if model.startswith(_SYSTEM_AI_PREFIX) else f"{_SYSTEM_AI_PREFIX}{model}"


def exp_name(name: str, run_id: str | None = None) -> str:
    """The temporary clone of model service ``name`` that candidates are evaluated against."""
    suffix = f"_{run_id}" if run_id else ""
    return f"{name}{EXPERIMENT_SUFFIX}{suffix}"


def service_path(name: str) -> str:
    """The REST path of model service ``name`` (``catalog.schema.name``)."""
    return f"{_MODEL_SERVICES_PATH}/{name}"


def _service_leaf(model: str) -> str:
    """``system.ai.databricks-claude-sonnet-4-5`` / ``claude-sonnet-4-5`` → ``claude-sonnet-4-5``."""
    for prefix in ("models/", _SYSTEM_AI_PREFIX, _FOUNDATION_PREFIX):
        model = model.removeprefix(prefix)
    return model


def foundation_model(client: WorkspaceClient | None, model: str) -> str:
    """The registered foundation model (``system.ai.databricks-<model>``) behind a ``system.ai`` model.

    Read from the ``system.ai.<model>`` service's own destination when a client is given (cached);
    otherwise, or if that lookup fails, the ``databricks-`` naming every system.ai chat service
    follows. Accepts either name form.
    """
    leaf = _service_leaf(model)
    if leaf in _foundation_cache:
        return _foundation_cache[leaf]
    resolved = f"{_SYSTEM_AI_PREFIX}{_FOUNDATION_PREFIX}{leaf}"
    if client is not None:
        try:
            raw = cast(
                "dict[str, Any]",
                client.api_client.do("GET", service_path(f"{_SYSTEM_AI_PREFIX}{leaf}")),
            )
            destinations = (raw.get("config") or {}).get("routing", {}).get("destinations") or []
            target = (
                (destinations[0].get("pay_per_token_config") or {}).get("model")
                if destinations
                else None
            )
            if target:
                resolved = target.removeprefix("models/")
        except Exception:  # noqa: BLE001 - fall back to the naming convention
            pass
        _foundation_cache[leaf] = resolved
    return resolved


def ppt_destination(foundation: str) -> dict[str, Any]:
    """A 100% pay-per-token destination for registered foundation model ``foundation``."""
    return {
        "name": "primary",
        "destination_type": _PPT_DESTINATION,
        "pay_per_token_config": {"model": f"models/{foundation.removeprefix('models/')}"},
        "traffic_percentage": 100,
    }


def destination_model(service: dict[str, Any]) -> str | None:
    """The model a model service routes to, as the ``system.ai.*`` model-service name agents use."""
    destinations = (service.get("config") or {}).get("routing", {}).get("destinations") or []
    if not destinations:
        return None
    model = (destinations[0].get("pay_per_token_config") or {}).get("model") or ""
    return system_ai_name(_service_leaf(model)) if model else None


def create_request(name: str, foundation: str, comment: str | None = None) -> tuple[dict, dict]:
    """The ``(query, body)`` that creates model service ``name`` routed to registered ``foundation``."""
    catalog, schema, leaf = name.split(".")
    body: dict[str, Any] = {"config": {"routing": {"destinations": [ppt_destination(foundation)]}}}
    if comment:
        body["comment"] = comment
    return {"parent": f"schemas/{catalog}.{schema}", "model_service_id": leaf}, body


def set_model_request(foundation: str) -> tuple[dict, dict]:
    """The ``(query, body)`` that repoints a model service to registered model ``foundation``."""
    return (
        {"update_mask": "config.routing.destinations"},
        {"config": {"routing": {"destinations": [ppt_destination(foundation)]}}},
    )


def grant_requests(name: str, principal: str) -> list[tuple[str, dict, bool]]:
    """``(path, body, required)`` for granting ``principal`` use of model service ``name``.

    EXECUTE on the service is required. USE CATALOG / USE SCHEMA on its parents are best-effort:
    the principal often holds them already (e.g. via ``account users``), and the caller may not
    manage the catalog.
    """
    catalog, schema, _ = name.split(".")

    def change(privilege: str) -> dict:
        return {"changes": [{"principal": principal, "add": [privilege]}]}

    return [
        (f"{_PERMISSIONS_PATH}/catalog/{catalog}", change("USE_CATALOG"), False),
        (f"{_PERMISSIONS_PATH}/schema/{catalog}.{schema}", change("USE_SCHEMA"), False),
        (f"{_PERMISSIONS_PATH}/model_service/{name}", change("EXECUTE"), True),
    ]


def can_execute(client: WorkspaceClient, model: str, principal: str) -> bool | None:
    """Whether ``principal`` holds EXECUTE on the registered model behind ``model``.

    A model service can only route to a model its owner can execute; UC rejects the repoint
    otherwise ("owner lacks EXECUTE on ... destination model(s)"), and workspaces often grant
    system.ai models individually. None when the grants can't be read (unknown, not denied).
    """
    foundation = foundation_model(client, model)
    try:
        raw = cast(
            "dict[str, Any]",
            client.api_client.do(
                "GET",
                f"/api/2.1/unity-catalog/effective-permissions/function/{foundation}",
                query={"principal": principal},
            ),
        )
    except Exception:  # noqa: BLE001 - unreadable grants aren't a denial
        return None
    privileges = {
        p.get("privilege")
        for assignment in (raw or {}).get("privilege_assignments") or []
        for p in assignment.get("privileges") or []
    }
    return "EXECUTE" in privileges or "ALL_PRIVILEGES" in privileges


def ensure_schema(client: WorkspaceClient, name: str) -> None:
    """Create the parent ``catalog.schema`` of model service ``name`` if it's missing.

    UC 404s a model-service create under a nonexistent schema with a bare "Resource not found".
    Reports a duplicate inconsistently (the typed AlreadyExists, or a 400 BadRequest saying
    "already exists"), so both are tolerated; anything else (no catalog, no permission) raises.
    """
    catalog, schema, _ = name.split(".")
    try:
        client.schemas.create(name=schema, catalog_name=catalog)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it's the duplicate case
        if "already exists" not in str(exc).lower():
            raise


def get_model(client: WorkspaceClient, name: str) -> str:
    """The ``system.ai.*`` model model service ``name`` routes to. Raises if it has none."""
    model = destination_model(
        cast("dict[str, Any]", client.api_client.do("GET", service_path(name)))
    )
    if model is None:
        raise ValueError(f"Model service {name!r} has no foundation-model destination")
    return model


def create(client: WorkspaceClient, name: str, model: str, comment: str | None = None) -> dict:
    """Create model service ``name`` routed to ``model``, creating its schema if missing."""
    ensure_schema(client, name)
    query, body = create_request(name, foundation_model(client, model), comment)
    return cast(
        "dict[str, Any]",
        client.api_client.do("POST", _MODEL_SERVICES_PATH, query=query, body=body),
    )


def set_model(client: WorkspaceClient, name: str, model: str) -> dict:
    """Repoint model service ``name`` to ``model``."""
    query, body = set_model_request(foundation_model(client, model))
    return cast(
        "dict[str, Any]",
        client.api_client.do("PATCH", service_path(name), query=query, body=body),
    )


def delete(client: WorkspaceClient, name: str) -> None:
    """Delete model service ``name``."""
    client.api_client.do("DELETE", service_path(name))


def delete_command(name: str) -> str:
    """A CLI command that deletes model service ``name`` (for manual-cleanup hints)."""
    return f"databricks api delete {service_path(name)}"
