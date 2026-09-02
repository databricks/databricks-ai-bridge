"""Discover a workspace's chat-capable serving endpoints.

Used by the demo chat app's model picker to offer the models an agent can be pointed at. Kept
framework-neutral (no agent SDK, no ``mlflow``) so it lives alongside the other neutral runtime
helpers and can be listed from either template's ``runtime/ui.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


def list_chat_model_endpoints(client: WorkspaceClient) -> list[str]:
    """Return the names of the workspace's chat-capable serving endpoints, sorted.

    Keeps endpoints whose task is a chat-completions task — foundation models, external models, and
    custom chat agents all report a task containing ``chat`` (e.g. ``llm/v1/chat``, ``agent/v2/chat``),
    while embeddings/completions endpoints do not — so the picker offers only models the agent's
    chat client can actually call. Endpoints with no task set are skipped.

    Raises whatever ``serving_endpoints.list()`` raises (e.g. a permission error); callers that want a
    graceful fallback to just the agent's default model should catch it.
    """
    names: list[str] = []
    for endpoint in client.serving_endpoints.list():
        name = endpoint.name
        task = (endpoint.task or "").lower()
        if name and "chat" in task:
            names.append(name)
    return sorted(names)
