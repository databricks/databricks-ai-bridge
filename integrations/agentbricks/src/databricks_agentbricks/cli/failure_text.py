"""Bound error text and scrub credentials, paths, and emails before regional telemetry."""

from __future__ import annotations

import re

MAX_FAILURE_BYTES = 4096
_MAX_INPUT_CHARS = 65536
_TRUNCATED = "\n[truncated]"
_REDACTED = "[REDACTED]"
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_HEADER = re.compile(
    r"(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)\b[\"']?\s*[:=][^\n]*"
)
_ASSIGNMENT = re.compile(
    r"(?ix)(?<![\w-])(?=[\w-]*(?:token|password|passwd|secret|api[_-]?key|access[_-]?key|"
    r"credential|signature|sig))[\w-]+[\"']?\s*[:=]\s*"
)
_SCALAR_VALUE = re.compile(
    r"""(?:"(?:\\[\s\S]|[^"\\])*(?:"|\\?\Z)|'(?:\\[\s\S]|[^'\\])*(?:'|\\?\Z)|[^\s,;&}\]]+)"""
)
_AUTH_SCHEME = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9+/_.=~-]+")
_URL_USERINFO = re.compile(r"(?i)(?<![a-z0-9+.-])([a-z][a-z0-9+.-]*://)[^\s/@]+@")
_URL = re.compile(r"(?i)https?://[^\s'\"<>]+")
_TOKEN = re.compile(
    r"\b(?:dapi[a-fA-F0-9]{16,}|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b"
)
_JWT_CANDIDATE = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)*")
# Follow the DABs path/email minimization pattern in databricks/cli libs/telemetry/scrub.go.
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s:,\"'/\\]+[\\/][^\s:,\"']+")
_PATHS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|[\s:,\"'])(/Workspace/[^\s:,\"']+)",
        r"(^|[\s:,\"'])(~?/[^\s:,\"'/]+/[^\s:,\"']+)",
        r"(^|[\s:,\"'])((?:\.\.?|\.[a-zA-Z][^\s:,\"'/]*)/[^\s:,\"']+)",
    )
)
_PATH_CANDIDATE = re.compile(r"(^|[\s:,\"'])([a-zA-Z0-9_][^\s:,\"']*)")
_EMAIL = re.compile(r"(?<![a-zA-Z0-9._%+\-])[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _container_end(value: str, start: int) -> int:
    closers: list[str] = []
    quote = ""
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if escaped:
            escaped = False
        elif quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]":
            if char != closers.pop():
                return len(value)
            if not closers:
                return index + 1
    return len(value)


def _scrub_assignments(value: str) -> str:
    parts: list[str] = []
    end = 0
    while match := _ASSIGNMENT.search(value, end):
        parts.extend((value[end : match.start()], _REDACTED))
        end = match.end()
        if end < len(value) and value[end] in "{[":
            # Redact the whole credential container, including unnamed values and nested keys.
            end = _container_end(value, end)
        elif scalar := _SCALAR_VALUE.match(value, end):
            end = scalar.end()
    return "".join((*parts, value[end:]))


def _scrub_relative_path(match: re.Match[str]) -> str:
    _, slash, remainder = match[2].partition("/")
    if slash and re.search(r"\.[a-zA-Z]", remainder):
        return match[1] + "[REDACTED_PATH]"
    return match[0]


def _scrub_url_query(match: re.Match[str]) -> str:
    parts = re.split(r"[?#]", match[0], maxsplit=1)
    return parts[0] + "?[REDACTED]" if len(parts) == 2 else parts[0]


def _scrub_jwt(match: re.Match[str]) -> str:
    segments = match[0].split(".")
    for index, segment in enumerate(segments[:-1]):
        if not segments[index + 1]:
            continue
        header = re.search(r"\beyJ[A-Za-z0-9_-]+", segment)
        if header is not None:
            prefix = ".".join([*segments[:index], segment[: header.start()]])
            return prefix + _REDACTED
    return match[0]


def scrub_failure_text(value: str) -> str:
    """Best-effort scrubbing; the result remains sensitive customer content."""
    truncated = len(value) > _MAX_INPUT_CHARS
    value = _ANSI.sub("", value[:_MAX_INPUT_CHARS])
    value = "".join(c for c in value if c in "\n\t" or (c.isprintable() and c != "\x7f"))
    value = _PRIVATE_KEY.sub(_REDACTED, value)
    value = _HEADER.sub(_REDACTED, value)
    value = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    value = _URL.sub(_scrub_url_query, value)
    value = _scrub_assignments(value)
    value = _AUTH_SCHEME.sub(_REDACTED, value)
    value = _TOKEN.sub(_REDACTED, value)
    value = _JWT_CANDIDATE.sub(_scrub_jwt, value)
    value = _WINDOWS_PATH.sub("[REDACTED_PATH]", value)
    for pattern in _PATHS:
        value = pattern.sub(r"\1[REDACTED_PATH]", value)
    value = _PATH_CANDIDATE.sub(_scrub_relative_path, value)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value).strip()
    if truncated:
        value += _TRUNCATED
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) > MAX_FAILURE_BYTES:
        marker = _TRUNCATED + "\n"
        budget = (MAX_FAILURE_BYTES - len(marker)) // 2
        return (
            encoded[:budget].decode("utf-8", errors="ignore")
            + marker
            + encoded[-budget:].decode("utf-8", errors="ignore")
        )
    return value
