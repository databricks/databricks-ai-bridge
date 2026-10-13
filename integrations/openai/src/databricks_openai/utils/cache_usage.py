"""Normalize provider cache usage for OpenAI-compatible consumers."""

from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk


def normalize_cache_usage(data: object, cast_to: type) -> object:
    """Expose Claude cache counters to OpenAI consumers without changing token totals.

    Databricks already includes cache reads and writes in ``prompt_tokens``. Retain the native
    counters as well as OpenAI-style details, for both completions and streaming chunks.
    """
    if cast_to not in (ChatCompletion, ChatCompletionChunk) or not isinstance(data, dict):
        return data
    data = cast(dict[str, Any], data)
    usage = data.get("usage")
    if not isinstance(usage, dict) or not any(
        key in usage for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    ):
        return data
    details = dict(usage.get("prompt_tokens_details") or {})
    details["cached_tokens"] = usage.get("cache_read_input_tokens") or 0
    details["cache_write_tokens"] = usage.get("cache_creation_input_tokens") or 0
    return {**data, "usage": {**usage, "prompt_tokens_details": details}}
