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

# Models skipped from Chat Completions tests (behavioral issues, not API limitations)
CHAT_SKIP_MODELS = {
    "databricks-gpt-5-nano",  # too small for reliable tool calling
    "databricks-gpt-oss-20b",  # hallucinates tool names in agent loop
    "databricks-gpt-oss-120b",  # hallucinates tool names in agent loop
    "databricks-llama-4-maverick",  # hallucinates tool names in agent loop
    "databricks-gemini-3-flash",  # requires thought_signature on function calls
    "databricks-gemini-3-pro",  # requires thought_signature on function calls
    "databricks-gemini-3-1-pro",  # requires thought_signature on function calls
}

# Additional models skipped only in LangChain Chat Completions tests
LANGCHAIN_CHAT_SKIP_MODELS = CHAT_SKIP_MODELS | {
    "databricks-gemma-3-12b",  # outputs raw tool call text instead of executing tools
}

# Responses API only models — skip from Chat Completions, include in Responses API tests
RESPONSES_ONLY_MODELS = {
    "databricks-gpt-5-1-codex-max",
    "databricks-gpt-5-1-codex-mini",
    "databricks-gpt-5-2-codex",
    "databricks-gpt-5-3-codex",
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
FALLBACK_CHAT_MODELS = [
    "databricks-claude-sonnet-4-6",
    "databricks-claude-opus-4-6",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-5-2",
    "databricks-gpt-5-1",
    "databricks-qwen3-next-80b-a3b-instruct",
]

FALLBACK_RESPONSES_MODELS = [
    "databricks-gpt-5-2",
    "databricks-gpt-5-1",
    "databricks-gpt-5-2-codex",
    "databricks-gpt-5-3-codex",
]


def has_function_calling(w: WorkspaceClient, endpoint_name: str) -> bool:
    """Check if an endpoint supports function calling via the capabilities API."""
    try:
        resp = w.api_client.do("GET", f"/api/2.0/serving-endpoints/{endpoint_name}")
        return resp.get("capabilities", {}).get("function_calling", False)  # type: ignore[invalid-assignment]
    except Exception:
        return False


def _supports_responses_api(name: str) -> bool:
    """Only OpenAI GPT models (not OSS) support the Responses API on Databricks FMAPI."""
    return "gpt" in name.lower() and "oss" not in name.lower()


def discover_chat_models(skip_models: set[str]) -> list[str]:
    """Discover FMAPI models for Chat Completions tests.

    Excludes Responses API-only models (codex) and models in skip_models.
    """
    try:
        w = WorkspaceClient()
        endpoints = list(w.serving_endpoints.list())
    except Exception as exc:
        log.warning("Could not discover FMAPI models, using fallback list: %s", exc)
        return FALLBACK_CHAT_MODELS

    chat_endpoints = [
        e
        for e in endpoints
        if e.name and e.name.startswith("databricks-") and e.task == "llm/v1/chat"
    ]

    models = []
    for e in sorted(chat_endpoints, key=lambda e: e.name or ""):
        name = e.name or ""
        if name in RESPONSES_ONLY_MODELS:
            log.info("Skipping %s from chat tests: Responses API only", name)
            continue
        if not has_function_calling(w, name):
            log.info("Skipping %s: does not support function calling", name)
            continue
        if name in skip_models:
            log.info("Skipping %s: in skip list", name)
            continue
        models.append(name)

    log.info("Discovered %d chat completions models", len(models))
    return models


def discover_responses_models() -> list[str]:
    """Discover FMAPI models for Responses API tests.

    Only GPT models support the Responses API on Databricks.
    Includes both codex (Responses-only) and regular GPT models.
    """
    try:
        w = WorkspaceClient()
        endpoints = list(w.serving_endpoints.list())
    except Exception as exc:
        log.warning("Could not discover FMAPI models, using fallback list: %s", exc)
        return FALLBACK_RESPONSES_MODELS

    chat_endpoints = [
        e
        for e in endpoints
        if e.name and e.name.startswith("databricks-") and e.task == "llm/v1/chat"
    ]

    models = []
    for e in sorted(chat_endpoints, key=lambda e: e.name or ""):
        name = e.name or ""
        if not _supports_responses_api(name):
            continue
        if not has_function_calling(w, name):
            log.info("Skipping %s: does not support function calling", name)
            continue
        # Skip nano — too small for reliable tool calling in any API
        if "nano" in name:
            continue
        models.append(name)

    log.info("Discovered %d responses API models", len(models))
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
