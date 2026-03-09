"""Shared test utilities for FMAPI tool calling integration tests.

Used by both databricks-openai and databricks-langchain test suites.
"""

from __future__ import annotations

import logging

from databricks.sdk import WorkspaceClient

log = logging.getLogger(__name__)

# Max retries for flaky models (e.g. transient FMAPI errors, model non-determinism)
MAX_RETRIES = 3

# Default max_tokens for test requests
DEFAULT_MAX_TOKENS = 200

# Models shared across both OpenAI and LangChain skip lists
COMMON_SKIP_MODELS = {
    "databricks-gpt-5-nano",  # too small for reliable tool calling
    "databricks-gpt-oss-20b",  # hallucinates tool names in agent loop
    "databricks-gpt-oss-120b",  # hallucinates tool names in agent loop
    "databricks-llama-4-maverick",  # hallucinates tool names in agent loop
    "databricks-gemini-3-flash",  # requires thought_signature on function calls
    "databricks-gemini-3-pro",  # requires thought_signature on function calls
    "databricks-gemini-3-1-pro",  # requires thought_signature on function calls
    "databricks-gemini-2-5-pro",  # returns list content that breaks Agents SDK parsing
    "databricks-gemini-2-5-flash",  # returns list content that breaks Agents SDK parsing
    "databricks-gpt-5-1-codex-max",  # Responses API only, no Chat Completions support
    "databricks-gpt-5-1-codex-mini",  # Responses API only, no Chat Completions support
    "databricks-gpt-5-2-codex",  # Responses API only, no Chat Completions support
    "databricks-gpt-5-3-codex",  # Responses API only, no Chat Completions support
    "databricks-gpt-5-4",  # Requires /v1/responses for tool calling, not /v1/chat/completions
    "databricks-gemini-3-1-flash-lite",  # Requires thought_signature on function calls
}

# Additional models skipped only in LangChain tests
LANGCHAIN_SKIP_MODELS = COMMON_SKIP_MODELS | {
    "databricks-gemma-3-12b",  # outputs raw tool call text instead of executing tools
}

# Reasoning models consume reasoning tokens from the max_tokens budget.
# Gemini 2.5 Pro needs 200-600 reasoning tokens with 2 tools, so 200 is too small.
MODEL_MAX_TOKENS: dict[str, int] = {
    "databricks-gemini-2-5-pro": 1000,
}


def max_tokens_for_model(model: str) -> int:
    """Return appropriate max_tokens for a model, accounting for reasoning token overhead."""
    return MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)


# Fallback list if dynamic discovery fails (e.g. auth not configured at collection time)
FALLBACK_MODELS = [
    "databricks-claude-sonnet-4-6",
    "databricks-claude-opus-4-6",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-5-2",
    "databricks-gpt-5-1",
    "databricks-qwen3-next-80b-a3b-instruct",
]


def has_function_calling(w: WorkspaceClient, endpoint_name: str) -> bool:
    """Check if an endpoint supports function calling via the capabilities API."""
    try:
        resp = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint_name}")
        return resp.get("capabilities", {}).get("function_calling", False)  # type: ignore[invalid-assignment]
    except Exception:
        return False


def discover_foundation_models(skip_models: set[str]) -> list[str]:
    """Discover all FMAPI chat models that support tool calling.

    1. List all serving endpoints with databricks- prefix and llm/v1/chat task
    2. Check capabilities.function_calling via the serving-endpoints API
    3. Models in skip_models are excluded entirely
    """
    try:
        w = WorkspaceClient()
        endpoints = list(w.serving_endpoints.list())
    except Exception as exc:
        log.warning("Could not discover FMAPI models, using fallback list: %s", exc)
        return FALLBACK_MODELS

    chat_endpoints = [
        e
        for e in endpoints
        if e.name and e.name.startswith("databricks-") and e.task == "llm/v1/chat"
    ]

    models = []
    for e in sorted(chat_endpoints, key=lambda e: e.name or ""):
        name = e.name or ""
        if not has_function_calling(w, name):
            log.info("Skipping %s: does not support function calling", name)
            continue
        if name in skip_models:
            log.info("Skipping %s: in skip list", name)
            continue
        models.append(name)

    log.info("Discovered %d FMAPI models with function calling support", len(models))
    return models


def retry(fn, retries=MAX_RETRIES):
    """Retry a test function up to `retries` times. Only fails if all attempts fail."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                log.warning("Attempt %d/%d failed: %s — retrying", attempt + 1, retries, exc)
    raise last_exc  # type: ignore[misc]


async def async_retry(fn, retries=MAX_RETRIES):
    """Retry an async test function up to `retries` times."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                log.warning("Attempt %d/%d failed: %s — retrying", attempt + 1, retries, exc)
    raise last_exc  # type: ignore[misc]
