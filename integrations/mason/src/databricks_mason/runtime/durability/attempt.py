"""Execution of one fenced durable attempt."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import cast

from databricks_mason.runtime.durability.types import (
    DurabilityStore,
    DurableExecutionContext,
    DurableExecutorFn,
    JsonObject,
    JsonValue,
)

logger = logging.getLogger(__name__)


def copy_json_value(value: JsonValue, name: str) -> JsonValue:
    """Return a detached JSON value or reject a value that cannot be persisted."""
    try:
        return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable") from exc


def _copy_json_object(value: JsonObject, name: str) -> JsonObject:
    copied = copy_json_value(value, name)
    if not isinstance(copied, dict):
        raise TypeError(f"{name} must be a JSON object")
    return copied


class AttemptRunner:
    """Claim, execute, heartbeat, and commit one durable attempt."""

    def __init__(
        self,
        execute_fn: DurableExecutorFn,
        *,
        durability_store: DurabilityStore,
        heartbeat_seconds: float,
        stale_seconds: float,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if stale_seconds <= heartbeat_seconds:
            raise ValueError("stale_seconds must be greater than heartbeat_seconds")
        self._execute_fn = execute_fn
        self._durability_store = durability_store
        self._heartbeat_seconds = heartbeat_seconds
        self._stale_seconds = stale_seconds

    async def run(self, execution_id: str) -> None:
        """Claim and run one eligible attempt, if this worker wins ownership."""
        try:
            claimed = await self._durability_store.claim(execution_id, self._stale_seconds)
        except Exception:
            logger.exception("Failed to claim durable execution: %s", execution_id)
            return
        if claimed is None:
            return

        heartbeat = asyncio.create_task(
            self._heartbeat_loop(execution_id, claimed.attempt),
            name=f"durable-heartbeat-{execution_id}-{claimed.attempt}",
        )
        try:

            async def emit(event: JsonObject) -> int:
                sequence_number = await self._durability_store.append_event(
                    execution_id,
                    claimed.attempt,
                    _copy_json_object(event, "event"),
                )
                if sequence_number is None:
                    raise RuntimeError(
                        f"execution {execution_id!r} no longer owns attempt {claimed.attempt}"
                    )
                return sequence_number

            response = await self._execute_fn(
                copy.deepcopy(claimed.request),
                DurableExecutionContext(
                    execution_id=execution_id,
                    attempt=claimed.attempt,
                    _emit=emit,
                ),
            )
            response = copy_json_value(response, "executor response")
            completed = await self._durability_store.complete(
                execution_id,
                claimed.attempt,
                response,
            )
            if not completed:
                logger.info(
                    "Skipped completion after durability ownership changed: %s attempt=%d",
                    execution_id,
                    claimed.attempt,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Durable execution failed: %s attempt=%d",
                execution_id,
                claimed.attempt,
            )
            try:
                await self._durability_store.fail(execution_id, claimed.attempt)
            except Exception:
                logger.exception(
                    "Failed to persist durable failure: %s attempt=%d",
                    execution_id,
                    claimed.attempt,
                )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat_loop(self, execution_id: str, attempt: int) -> None:
        while True:
            try:
                owns_attempt = await self._durability_store.heartbeat(execution_id, attempt)
            except Exception:
                logger.warning(
                    "Durable heartbeat failed: %s attempt=%d",
                    execution_id,
                    attempt,
                    exc_info=True,
                )
                await asyncio.sleep(self._heartbeat_seconds)
                continue
            if not owns_attempt:
                return
            await asyncio.sleep(self._heartbeat_seconds)
