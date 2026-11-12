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
    r"(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)\b[\"']?\s*[:=][^\S\n]*"
)
_CREDENTIAL_NAME = (
    r"(?:(?=[\w-]*(?:token|password|passwd|secret|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential|signature))[\w-]+|sig)"
)
_ASSIGNMENT = re.compile(rf"(?ix)(?<![\w-]){_CREDENTIAL_NAME}[\"']?\s*[:=][^\S\n]*")
_OPTION = re.compile(rf"(?ix)(?<![\w-])--?{_CREDENTIAL_NAME}[\"']?(?:\s*,\s*|(?=\s)[^\S\n]*)")
# Encoded text may hide credential names, so omit it without trying to decode it.
_ENCODED = re.compile(r"""\\(?:["'\\]|[uUxX][0-9a-fA-F])""")
_AUTH_SCHEME = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9+/_.=~-]+")
_URL_USERINFO = re.compile(r"""(?i)(?<![a-z0-9+.-])([a-z][a-z0-9+.-]*://)[^\s/?#"'<>]+@""")
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
        r"(^|[\s:=,\"'])(/Workspace/[^\s:,\"']+)",
        r"(^|[\s:=,\"'])(~?/[^\s:,\"'/]+/[^\s:,\"']+)",
        r"(^|[\s:=,\"'])((?:\.\.?|\.[a-zA-Z][^\s:,\"'/]*)/[^\s:,\"']+)",
    )
)
_PATH_CANDIDATE = re.compile(r"(^|[\s:=,\"'])([a-zA-Z0-9_][^\s:,\"']*)")
_EMAIL = re.compile(r"(?<![a-zA-Z0-9._%+\-])[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


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
    # Free-form credential values have no reliable end boundary; retain only preceding text.
    for pattern in (_ENCODED, _HEADER, _ASSIGNMENT, _OPTION):
        if match := pattern.search(value):
            value = value[: match.start()] + _REDACTED
    value = _PRIVATE_KEY.sub(_REDACTED, value)
    value = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    value = _URL.sub(_scrub_url_query, value)
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
