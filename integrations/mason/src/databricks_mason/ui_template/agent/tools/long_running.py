"""Deterministic long-running tools used by the Mason stop/start demo."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool  # ty: ignore[unresolved-import]

_PROCESS_ID = uuid.uuid4().hex[:12]
_STEP_SECONDS_ENV = "MASON_DEMO_TOOL_STEP_SECONDS"


def process_id() -> str:
    """Return the identifier shared by this process's UI and tool outputs."""
    return _PROCESS_ID


def step_seconds() -> float:
    """Duration of each demo step, configurable for tests and local demos."""
    return max(0.0, float(os.getenv(_STEP_SECONDS_ENV, "6")))


async def _run_step(step: int) -> dict[str, Any]:
    await asyncio.sleep(step_seconds())
    name = f"tool_step_{step}"
    return {
        "tool": name,
        "output": f"{name} completed successfully",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": process_id(),
    }


@tool
async def tool_step_1() -> dict[str, Any]:
    """Run the first long-running demo step."""
    return await _run_step(1)


@tool
async def tool_step_2() -> dict[str, Any]:
    """Run the second long-running demo step."""
    return await _run_step(2)


@tool
async def tool_step_3() -> dict[str, Any]:
    """Run the third long-running demo step."""
    return await _run_step(3)


@tool
async def tool_step_4() -> dict[str, Any]:
    """Run the fourth long-running demo step."""
    return await _run_step(4)
