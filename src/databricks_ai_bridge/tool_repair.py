"""Shared orphan-tool-call repair logic.

``sanitize_tool_items`` walks a list of Responses-API-style items and
reconciles orphan / duplicate ``function_call`` / ``function_call_output``
items. Used by:

* the server-side input sanitizer in :mod:`...long_running.server`, which
  runs on every request before the handler is invoked; and
* the OpenAI :class:`AsyncDatabricksSession` ``get_items`` auto-repair,
  which returns protocol-valid items without touching the underlying DB.

The LangChain checkpointer has its own repair path
(``_build_tool_resume_repair``) that operates on ``AIMessage`` /
``ToolMessage`` shapes rather than the dict items here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Default body for the synthetic ``function_call_output`` injected when a
#: prior attempt's tool call has no matching output (e.g. the pod was killed
#: between emitting the call and its result). Shared between the server-side
#: input sanitizer and integration-side read-time repair paths so the user-
#: visible text stays consistent across the durable-resume contract.
DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT = (
    "[INTERRUPTED] This tool call did not complete due to a server "
    "interruption, so no result is available. Other tool calls in the "
    "conversation history completed normally and their results remain valid. "
    "If the information is still needed, re-invoking only this specific tool "
    "is usually sufficient."
)


def _default_item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def sanitize_tool_items(
    items: list[Any],
    synthetic_output: str = DEFAULT_SYNTHETIC_INTERRUPTED_OUTPUT,
    *,
    item_get: Callable[[Any, str], Any] = _default_item_get,
    log_prefix: str = "[durable] items sanitized",
) -> list[Any]:
    """Return a protocol-valid view of ``items``.

    In order:

    * drops duplicate ``function_call`` items by ``call_id``;
    * drops duplicate or orphan ``function_call_output`` items (no matching
      ``function_call`` anywhere in the list);
    * injects a synthetic ``function_call_output`` immediately after any
      ``function_call`` that has no output in the list.

    Also recognises chat-completions-shape ``{role: assistant, tool_calls:
    [...]}`` items as declaring call_ids, so mixed-shape histories don't
    trip the orphan check.

    Returns the caller's ``items`` reference unchanged on the happy path so
    downstream can skip any re-persistence cheaply.

    The ``synthetic_output`` text is passed in by the caller — each caller
    owns its own copy of the string so product decisions about wording
    stay scoped to the durable-resume path they belong to.

    ``item_get`` lets session-style objects (ORM rows with attribute
    access) reuse this walker; defaults to plain dict ``.get``.
    """
    if not items:
        return items

    declared_call_ids: set[str] = set()
    call_ids_with_output: set[str] = set()
    for item in items:
        t = item_get(item, "type")
        cid = item_get(item, "call_id")
        if t == "function_call" and cid:
            declared_call_ids.add(cid)
        if t == "function_call_output" and cid:
            call_ids_with_output.add(cid)
        # Chat-completions shape: assistant message with tool_calls.
        if item_get(item, "role") == "assistant":
            tool_calls = item_get(item, "tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id") or (tc.get("function") or {}).get("id")
                    if tc_id:
                        declared_call_ids.add(tc_id)

    sanitized: list[Any] = []
    seen_calls: set[str] = set()
    seen_outputs: set[str] = set()
    injected = 0
    dropped_orphan_outputs = 0
    dropped_duplicates = 0

    for item in items:
        t = item_get(item, "type")
        cid = item_get(item, "call_id")
        if t == "function_call" and cid:
            if cid in seen_calls:
                dropped_duplicates += 1
                continue
            seen_calls.add(cid)
            sanitized.append(item)
            if cid not in call_ids_with_output:
                sanitized.append(
                    {
                        "type": "function_call_output",
                        "call_id": cid,
                        "output": synthetic_output,
                    }
                )
                injected += 1
        elif t == "function_call_output" and cid:
            if cid in seen_outputs:
                dropped_duplicates += 1
                continue
            if cid not in declared_call_ids:
                dropped_orphan_outputs += 1
                continue
            seen_outputs.add(cid)
            sanitized.append(item)
        else:
            sanitized.append(item)

    if not (injected or dropped_orphan_outputs or dropped_duplicates):
        # Happy path: hand back the original list so callers can skip
        # re-persistence by identity comparison (``sanitized is items``).
        return items

    logger.info(
        "%s: injected=%d dropped_orphan_outputs=%d dropped_duplicates=%d original=%d final=%d",
        log_prefix,
        injected,
        dropped_orphan_outputs,
        dropped_duplicates,
        len(items),
        len(sanitized),
    )
    return sanitized
