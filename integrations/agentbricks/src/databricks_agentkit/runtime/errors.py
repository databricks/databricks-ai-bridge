"""Short, credential-free descriptions of failed invocations.

The full traceback stays in the server log. Callers get the exception type, the upstream HTTP
status and error code when an upstream service rejected a call, and a redacted, truncated message.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from databricks_agentkit.runtime.types import JsonObject

MAX_MESSAGE_LENGTH = 300
_MAX_CODE_LENGTH = 100
_MAX_CHAIN = 32
_REDACTED = "[REDACTED]"
_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
_MIN_ENV_SECRET_LENGTH = 8

# Databricks SDK error classes carry no status attribute; map them by class name.
_DATABRICKS_SDK_STATUS = {
    "BadRequest": 400,
    "Unauthenticated": 401,
    "PermissionDenied": 403,
    "NotFound": 404,
    "ResourceConflict": 409,
    "TooManyRequests": 429,
    "Cancelled": 499,
    "InternalError": 500,
    "NotImplemented": 501,
    "TemporarilyUnavailable": 503,
    "DeadlineExceeded": 504,
}
# Matched against the whole MRO, so subclasses such as httpx.ConnectTimeout count too.
_TIMEOUT_CLASS_NAMES = frozenset(
    {"APITimeoutError", "TimeoutException", "Timeout", "ReadTimeout", "DeadlineExceeded"}
)
_TIMEOUT_STATUSES = frozenset({408, 504})
_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bBearer\s+[^\s'\",]+"), "Bearer " + _REDACTED),
    (re.compile(r"\bdapi[0-9A-Za-z-]{8,}"), _REDACTED),
    (re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+"), _REDACTED),
    (
        re.compile(
            r"(?i)\b((?:access_|refresh_|id_)?token|secret|password|api[_-]?key)"
            r"([\"']?\s*[=:]\s*[\"']?)[^\s&\"',}]+"
        ),
        r"\1\2" + _REDACTED,
    ),
)


@dataclass(frozen=True)
class InvocationFailure:
    """The caller-facing error object and HTTP status for one failed invocation."""

    error: JsonObject
    status_code: int

    @property
    def retry_after(self) -> int | None:
        retry_after = self.error.get("retry_after")
        return retry_after if isinstance(retry_after, int) else None

    def response(self) -> JsonObject:
        """The value persisted as the failed invocation's response."""
        return {"error": dict(self.error), "status_code": self.status_code, "source": "agent"}


class InvocationError(RuntimeError):
    """An agent failure that has already been described and redacted."""

    def __init__(self, failure: InvocationFailure) -> None:
        message = failure.error.get("message")
        super().__init__(message if isinstance(message, str) else "agent invocation failed")
        self.failure = failure


def describe_failure(exc: BaseException, secrets: Iterable[str | None] = ()) -> InvocationFailure:
    """Describe ``exc`` without credentials or tracebacks."""
    if isinstance(exc, InvocationError):
        return exc.failure
    known_secrets = _known_secrets(secrets)
    upstream: BaseException | None = None
    upstream_status: int | None = None
    timed_out = False
    for candidate in _exception_chain(exc):
        upstream_status = _http_status(candidate)
        timed_out = any(base.__name__ in _TIMEOUT_CLASS_NAMES for base in type(candidate).__mro__)
        if upstream_status is not None or timed_out or _is_databricks_sdk_error(candidate):
            upstream = candidate
            break

    source = upstream or exc
    error: JsonObject = {"code": "AGENT_ERROR", "type": _type_name(source)}
    status_code = 500
    if upstream is not None:
        if upstream_status == 429:
            error["code"], status_code = "UPSTREAM_RATE_LIMITED", 429
        elif timed_out or upstream_status in _TIMEOUT_STATUSES:
            error["code"], status_code = "UPSTREAM_TIMEOUT", 504
        else:
            error["code"], status_code = "UPSTREAM_ERROR", 502
    body = _upstream_body(source) if upstream is not None else None
    error["message"] = redact(_message(source, body), known_secrets) or type(source).__name__
    if upstream_status is not None:
        error["upstream_status"] = upstream_status
    upstream_code = _upstream_code(source, body) if upstream is not None else None
    if upstream_code is not None:
        error["upstream_code"] = redact(upstream_code, known_secrets, limit=_MAX_CODE_LENGTH)
    retry_after = _retry_after(source) if status_code == 429 else None
    if retry_after is not None:
        error["retry_after"] = retry_after
    return InvocationFailure(error=error, status_code=status_code)


def redact(message: str, secrets: Iterable[str] = (), *, limit: int = MAX_MESSAGE_LENGTH) -> str:
    """Remove tracebacks and credentials, collapse to one line, and truncate."""
    message = message.split("Traceback (most recent call last)", 1)[0]
    longest_first: list[str] = sorted(set(secrets), key=lambda value: -len(value))
    for secret in longest_first:
        message = message.replace(secret, _REDACTED)
    for pattern, replacement in _SECRET_PATTERNS:
        message = pattern.sub(replacement, message)
    message = " ".join(message.split())
    if len(message) > limit:
        message = message[: limit - 1].rstrip() + "…"
    return message


def _known_secrets(secrets: Iterable[str | None]) -> list[str]:
    values = [secret for secret in secrets if secret]
    for name, value in os.environ.items():
        if len(value) >= _MIN_ENV_SECRET_LENGTH and any(
            marker in name.upper() for marker in _SECRET_ENV_MARKERS
        ):
            values.append(value)
    return values


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc``, grouped exceptions, then causes and contexts, depth first."""
    pending = [exc]
    seen: set[int] = set()
    while pending and len(seen) < _MAX_CHAIN:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        linked: list[BaseException] = []
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (list, tuple)):
            linked.extend(item for item in grouped if isinstance(item, BaseException))
        if current.__cause__ is not None:
            linked.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            linked.append(current.__context__)
        pending.extend(reversed(linked))


def _http_status(exc: BaseException) -> int | None:
    candidates = [
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "http_status", None),
    ]
    if _is_databricks_sdk_error(exc):
        candidates.extend(_DATABRICKS_SDK_STATUS.get(base.__name__) for base in type(exc).__mro__)
    for status in candidates:
        if isinstance(status, int) and not isinstance(status, bool) and 400 <= status <= 599:
            return status
    return None


def _is_databricks_sdk_error(exc: BaseException) -> bool:
    return any(
        base.__name__ == "DatabricksError" and base.__module__.startswith("databricks.sdk")
        for base in type(exc).__mro__
    )


def _type_name(exc: BaseException) -> str:
    cls = type(exc)
    module = ".".join(part for part in cls.__module__.split(".") if not part.startswith("_"))
    if not module or module == "builtins":
        return cls.__qualname__
    return f"{module}.{cls.__qualname__}"


def _upstream_body(exc: BaseException) -> Mapping[str, object] | None:
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        json_body = getattr(response, "json", None)
        if callable(json_body):
            try:
                body = json_body()
            except Exception:
                body = None
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        return body["error"]
    return body if isinstance(body, Mapping) else None


def _message(exc: BaseException, body: Mapping[str, object] | None) -> str:
    # openai's str(exc) is "Error code: 429 - {...}"; prefer the upstream's own message.
    for key in ("message", "detail", "error_description"):
        value = body.get(key) if body is not None else None
        if isinstance(value, str) and value.strip():
            return value
    return str(exc)


def _upstream_code(exc: BaseException, body: Mapping[str, object] | None) -> str | None:
    candidates = [
        body.get("error_code") if body is not None else None,
        body.get("code") if body is not None else None,
        getattr(exc, "error_code", None),
        getattr(exc, "code", None),
    ]
    for code in candidates:
        if isinstance(code, str) and code.strip():
            return code
    return None


def _retry_after(exc: BaseException) -> int | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None and hasattr(headers, "get"):
        try:
            milliseconds = headers.get("retry-after-ms")
            if milliseconds is not None:
                return max(0, math.ceil(float(milliseconds) / 1000))
            seconds = headers.get("retry-after")
            if seconds is not None:
                return max(0, int(str(seconds).strip()))
        except (TypeError, ValueError, OverflowError):
            pass  # HTTP-date values are not forwarded.
    seconds = getattr(exc, "retry_after_secs", None)
    if isinstance(seconds, int) and not isinstance(seconds, bool) and seconds >= 0:
        return seconds
    return None
