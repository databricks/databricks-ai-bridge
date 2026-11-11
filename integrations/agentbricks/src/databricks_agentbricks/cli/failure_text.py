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
    r"(?ix)\b[\w-]*(?:token|password|passwd|secret|api[_-]?key|access[_-]?key|"
    r"credential|signature|sig)\b[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*(?:\"|$)|'[^']*(?:'|$)|[^\s,;&}\]]+)"
)
_AUTH_SCHEME = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9+/_.=~-]+")
_URL_USERINFO = re.compile(r"(?i)(?<![a-z0-9+.-])([a-z][a-z0-9+.-]*://)[^\s/@]+@")
_URL_QUERY = re.compile(r"(?i)(https?://[^\s?#'\"<>]+)[?#][^\s'\"<>]*")
_TOKEN = re.compile(
    r"\b(?:dapi[a-fA-F0-9]{16,}|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?)\b"
)
# Follow the DABs path/email minimization pattern in databricks/cli libs/telemetry/scrub.go.
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s:,\"'/\\]+[\\/][^\s:,\"']+")
_PATHS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|[\s:,\"'])(/Workspace/[^\s:,\"']+)",
        r"(^|[\s:,\"'])(~?/[^\s:,\"'/]+/[^\s:,\"']+)",
        r"(^|[\s:,\"'])((?:\.\.?|\.[a-zA-Z][^\s:,\"'/]*)/[^\s:,\"']+)",
        r"(^|[\s:,\"'])([a-zA-Z0-9_][^\s:,\"']*/[^\s:,\"']*\.[a-zA-Z][^\s:,\"']*)",
    )
)
_EMAIL = re.compile(r"(?<![a-zA-Z0-9._%+\-])[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def scrub_failure_text(value: str) -> str:
    """Best-effort scrubbing; the result remains sensitive customer content."""
    truncated = len(value) > _MAX_INPUT_CHARS
    value = _ANSI.sub("", value[:_MAX_INPUT_CHARS])
    value = "".join(c for c in value if c in "\n\t" or (c.isprintable() and c != "\x7f"))
    value = _PRIVATE_KEY.sub(_REDACTED, value)
    value = _HEADER.sub(_REDACTED, value)
    value = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    value = _URL_QUERY.sub(r"\1?[REDACTED]", value)
    value = _ASSIGNMENT.sub(_REDACTED, value)
    value = _AUTH_SCHEME.sub(_REDACTED, value)
    value = _TOKEN.sub(_REDACTED, value)
    value = _WINDOWS_PATH.sub("[REDACTED_PATH]", value)
    for pattern in _PATHS:
        value = pattern.sub(r"\1[REDACTED_PATH]", value)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value).strip()
    encoded = value.encode("utf-8", errors="replace")
    if truncated or len(encoded) > MAX_FAILURE_BYTES:
        return (
            encoded[: MAX_FAILURE_BYTES - len(_TRUNCATED)].decode("utf-8", errors="ignore")
            + _TRUNCATED
        )
    return value
