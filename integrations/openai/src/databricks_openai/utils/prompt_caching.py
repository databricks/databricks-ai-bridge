"""Claude cache controls for the final OpenAI-compatible Chat Completions request."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk

PromptCaching = Literal["auto", "off"]
PromptCacheProvider = Literal["anthropic"]


def validate_prompt_caching(policy: str, provider: str | None) -> None:
    if policy not in ("auto", "off"):
        raise ValueError("prompt_caching must be 'auto' or 'off'")
    if provider not in (None, "anthropic"):
        raise ValueError("prompt_cache_provider must be None or 'anthropic'")


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


def _has_cache_control(value: Any) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(
            _has_cache_control(v)
            for key, v in value.items()
            if key not in ("parameters", "custom_inputs")
        )
    if isinstance(value, (list, tuple)):
        return any(_has_cache_control(v) for v in value)
    return False


def _last_cacheable_block(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        content = [{"type": "text", "text": content}]
        message["content"] = content
    if isinstance(content, list):
        for block in reversed(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                if block["text"].strip():
                    return block
            elif block.get("type") == "image_url":
                return block
    return None


def prepare_chat_request(
    kwargs: dict[str, Any], policy: PromptCaching, provider: PromptCacheProvider | None
) -> dict[str, Any]:
    """Copy eligible Claude inputs before adaptations and add at most four cache points.

    System instructions cache the preceding tool definitions too. The final three user/tool
    request boundaries cache growing history, retaining previous boundaries even when a parallel
    tool batch exceeds Claude's automatic lookback window. Explicit caller controls take precedence
    over this policy, including controls supplied through ``extra_body``.
    """
    if policy != "auto":
        return kwargs
    extra_body = kwargs.get("extra_body")
    model = (
        extra_body.get("model", kwargs.get("model"))
        if isinstance(extra_body, dict)
        else kwargs.get("model")
    )
    if not (provider == "anthropic" or (isinstance(model, str) and "claude" in model.lower())):
        return kwargs

    kwargs = dict(kwargs)
    for key in ("messages", "tools"):
        value = kwargs.get(key)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            kwargs[key] = deepcopy(list(value))

    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, dict):
        extra_body = deepcopy(extra_body)
        kwargs["extra_body"] = extra_body
    else:
        extra_body = {}
    if any(_has_cache_control(kwargs.get(key)) for key in ("messages", "tools", "extra_body")):
        return kwargs

    messages = extra_body.get("messages", kwargs.get("messages"))
    if not isinstance(messages, list):
        return kwargs
    system = []
    boundaries = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        message = cast(dict[str, Any], message)
        role = message.get("role")
        if role in ("system", "developer"):
            system.append(message)
        elif role in ("user", "tool"):
            # Only the final result in a parallel batch is a model-request boundary.
            following = messages[index + 1] if index + 1 < len(messages) else None
            if role == "tool" and isinstance(following, dict) and following.get("role") == "tool":
                continue
            boundaries.append(message)

    candidates = system[-1:] + boundaries[-3:]
    for message in candidates:
        block = _last_cacheable_block(message)
        if block is not None:
            block["cache_control"] = {"type": "ephemeral"}
    return kwargs
