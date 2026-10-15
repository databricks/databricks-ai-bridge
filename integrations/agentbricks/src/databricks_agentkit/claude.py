"""Claude prompt caching for AgentKit applications.

Generic connectors remain pass-through clients. Templates opt into this provider adapter.
"""

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, cast

from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.utils.clients import AsyncDatabricksChat, AsyncDatabricksCompletions
from openai import AsyncOpenAI


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


def prepare_claude_request(kwargs: dict[str, Any], policy: str = "auto") -> dict[str, Any]:
    """Prepare a template's Claude request without changing its stored messages.

    Sonnet 4.6 uses native automatic history placement plus one stable system marker to protect
    instructions beyond the 20-block lookback. Other recognized Claude names use explicit points.
    Caller-supplied controls take precedence. This policy belongs to AgentKit templates, not
    the generic OpenAI or LangChain connectors.
    """
    if policy not in ("auto", "off"):
        raise ValueError("DATABRICKS_CLAUDE_PROMPT_CACHING must be auto or off")
    if policy == "off":
        return kwargs
    extra_body = kwargs.get("extra_body")
    model = (
        extra_body.get("model", kwargs.get("model"))
        if isinstance(extra_body, dict)
        else kwargs.get("model")
    )
    if not (isinstance(model, str) and "claude" in model.lower()):
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

    # Native automatic placement was verified on Sonnet 4.6. Older Claude endpoints reject
    # the top-level field, so keep explicit points for them. Aliases can supply their own
    # cache controls through the framework's existing extra_body pass-through.
    native = model in ("system.ai.claude-sonnet-4-6", "databricks-claude-sonnet-4-6")
    if native:
        kwargs["extra_body"] = {**extra_body, "cache_control": {"type": "ephemeral"}}
    candidates = system[-1:] if native else system[-1:] + boundaries[-3:]
    for message in candidates:
        block = _last_cacheable_block(message)
        if block is not None:
            block["cache_control"] = {"type": "ephemeral"}
    return kwargs


class _ClaudeCompletions(AsyncDatabricksCompletions):
    def __init__(self, client: AsyncDatabricksOpenAI, policy: str) -> None:
        super().__init__(client)
        self._policy = policy

    async def create(self, **kwargs):
        return await super().create(**prepare_claude_request(kwargs, self._policy))


class ClaudeCachingOpenAI(AsyncDatabricksOpenAI):
    """AgentKit client that opts into the template's Claude request policy.

    Authentication, routing, serialization, and usage accounting stay in the generic connector.
    The provider adapter and its policy survive OpenAI Agents' retry-client copies.
    """

    def __init__(
        self, *, cache_policy: str = "auto", _copy_workspace_client: Any = None, **kwargs: Any
    ) -> None:
        if cache_policy not in ("auto", "off"):
            raise ValueError("DATABRICKS_CLAUDE_PROMPT_CACHING must be auto or off")
        self._cache_policy = cache_policy
        if _copy_workspace_client is None:
            super().__init__(**kwargs)
        else:
            # OpenAI's public copy API supplies the already-authorized HTTP client and
            # resolved base URL. Reuse those options rather than creating new credentials.
            self._workspace_client = _copy_workspace_client
            AsyncOpenAI.__init__(self, **kwargs)

    @property
    def chat(self) -> AsyncDatabricksChat:
        if not hasattr(self, "_claude_chat"):
            self._claude_chat = AsyncDatabricksChat(self)
            self._claude_chat.completions = _ClaudeCompletions(self, self._cache_policy)
        return self._claude_chat

    def copy(self, **kwargs: Any) -> "ClaudeCachingOpenAI":
        extra = dict(kwargs.pop("_extra_kwargs", {}))
        extra.update(_copy_workspace_client=self._workspace_client, cache_policy=self._cache_policy)
        return super().copy(_extra_kwargs=extra, **kwargs)

    with_options = copy
