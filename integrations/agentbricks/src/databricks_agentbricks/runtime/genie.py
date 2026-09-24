"""Framework-neutral tools for a configured Genie Agent's Chat-mode conversation API."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from databricks_agentbricks.runtime.workspace import workspace_client

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


async def _run_in_thread(function: Callable[..., Any], *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise cancelled


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a 32-character lowercase hexadecimal ID.")
    return value


class GenieAgent:
    """Call one configured space without allowing the model to choose credentials or scope."""

    def __init__(
        self,
        space_id: str,
        *,
        wait_seconds: float = 120.0,
        auth: str = "app",
        workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
    ) -> None:
        self.space_id = _identifier(space_id, "space_id")
        if not 0 < wait_seconds <= 600:
            raise ValueError("wait_seconds must be greater than zero and at most 600.")
        if auth not in ("user", "app"):
            raise ValueError("auth must be 'user' or 'app'.")
        self.wait_seconds = wait_seconds
        self.auth = auth
        self.workspace_client_for = workspace_client_for

    def _client(self) -> WorkspaceClient:
        if self.workspace_client_for is not None:
            return self.workspace_client_for(self.auth)
        return workspace_client()

    async def ask(self, question: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Ask the configured Genie Agent a data question, or continue a conversation.

        Waits up to two minutes for an answer. Preserve returned attachments, SQL and deep_link.
        If timed_out is true and IDs are returned, call the matching poll tool; do not resubmit.
        If indeterminate_submission is true, submission may still complete but no message ID was
        received: do not resubmit automatically. NOT_SUBMITTED means setup timed out before sending
        the question. FAILED, CANCELLED and QUERY_RESULT_EXPIRED are
        terminal, not successful answers. Use query_result with a query attachment's attachment_id
        to read its rows. No workspace or space selection is needed.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty.")
        if conversation_id is not None:
            _identifier(conversation_id, "conversation_id")
        deadline = time.monotonic() + self.wait_seconds
        try:
            client = await asyncio.wait_for(
                _run_in_thread(self._client), timeout=deadline - time.monotonic()
            )
        except asyncio.TimeoutError:
            return self._submission_timeout(conversation_id, indeterminate=False)
        try:
            if conversation_id is None:
                operation = await asyncio.wait_for(
                    _run_in_thread(client.genie.start_conversation, self.space_id, question),
                    timeout=deadline - time.monotonic(),
                )
                conversation_id = operation.response.conversation_id
            else:
                operation = await asyncio.wait_for(
                    _run_in_thread(
                        client.genie.create_message, self.space_id, conversation_id, question
                    ),
                    timeout=deadline - time.monotonic(),
                )
        except asyncio.TimeoutError:
            return self._submission_timeout(conversation_id, indeterminate=True)
        return await self._wait(
            client, conversation_id, operation.response.message_id, deadline=deadline
        )

    def _submission_timeout(
        self, conversation_id: str | None, *, indeterminate: bool
    ) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            **({"conversation_id": conversation_id} if conversation_id is not None else {}),
            "status": "INDETERMINATE_SUBMISSION" if indeterminate else "NOT_SUBMITTED",
            "timed_out": True,
            "indeterminate_submission": indeterminate,
            "warning": (
                "Submission may still complete, but no message ID was received. "
                "Do not resubmit automatically."
                if indeterminate
                else "Client setup timed out before submission; the question was not submitted."
            ),
        }

    async def poll(self, conversation_id: str, message_id: str) -> dict[str, Any]:
        """Resume waiting for a previously submitted Genie Agent message without resubmitting it.

        Waits up to two minutes, polling with backoff. If timed_out is true, retain the IDs to poll
        again later. Terminal failures and expired results are returned explicitly, never rerun.
        """
        _identifier(conversation_id, "conversation_id")
        _identifier(message_id, "message_id")
        deadline = time.monotonic() + self.wait_seconds
        try:
            client = await asyncio.wait_for(
                _run_in_thread(self._client), timeout=deadline - time.monotonic()
            )
        except asyncio.TimeoutError:
            return {
                "space_id": self.space_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "status": "IN_PROGRESS",
                "timed_out": True,
            }
        return await self._wait(client, conversation_id, message_id, deadline=deadline)

    def _reference(self, client: Any, conversation_id: str, message_id: str) -> dict[str, Any]:
        _identifier(conversation_id, "conversation_id")
        _identifier(message_id, "message_id")
        return {
            "space_id": self.space_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "deep_link": f"{client.config.host.rstrip('/')}/genie/spaces/{self.space_id}"
            f"/conversations/{conversation_id}",
        }

    async def _wait(
        self, client: Any, conversation_id: str, message_id: str, *, deadline: float
    ) -> dict[str, Any]:
        reference = self._reference(client, conversation_id, message_id)
        delay = 2.0
        message: dict[str, Any] = {"status": "IN_PROGRESS"}
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                response = await asyncio.wait_for(
                    _run_in_thread(
                        client.genie.get_message, self.space_id, conversation_id, message_id
                    ),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            message = response.as_dict()
            if message.get("status") in _TERMINAL:
                return {**message, **reference}
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, 30.0)
        return {**message, **reference, "timed_out": True}

    async def query_result(
        self, conversation_id: str, message_id: str, attachment_id: str
    ) -> dict[str, Any]:
        """Read up to 100 rows and column schema for a Genie Agent query attachment.

        Use the conversation_id, message_id and query attachment's attachment_id from ask/poll.
        Check status before interpreting rows. When truncated is true, deep_link opens the full
        result in Databricks. This never re-executes an expired query or downloads external links.
        """
        _identifier(conversation_id, "conversation_id")
        _identifier(message_id, "message_id")
        _identifier(attachment_id, "attachment_id")
        client = await _run_in_thread(self._client)
        response = await _run_in_thread(
            client.genie.get_message_attachment_query_result,
            self.space_id,
            conversation_id,
            message_id,
            attachment_id,
        )
        statement = response.as_dict().get("statement_response") or {}
        manifest = statement.get("manifest") or {}
        result = statement.get("result") or {}
        rows = result.get("data_array") or []
        total = manifest.get("total_row_count")
        return {
            **self._reference(client, conversation_id, message_id),
            "attachment_id": attachment_id,
            "status": statement.get("status") or {"state": "UNKNOWN"},
            "columns": (manifest.get("schema") or {}).get("columns", []),
            "rows": rows[:100],
            "total_row_count": total,
            "truncated": bool(
                len(rows) > 100
                or manifest.get("truncated")
                or result.get("next_chunk_index") is not None
                or (total is not None and total > len(rows[:100]))
            ),
        }
