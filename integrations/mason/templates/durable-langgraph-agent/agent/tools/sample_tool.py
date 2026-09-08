"""Sample tools for ordinary and crash-recovery invocations."""

import asyncio
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@tool
async def wait_for_seconds(seconds: int) -> str:
    """Wait for a requested duration before continuing, up to five minutes."""
    if not 1 <= seconds <= 300:
        raise ValueError("seconds must be between 1 and 300")
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds."
