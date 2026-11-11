"""Bound error text and remove recognizable credentials before regional telemetry."""

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
_URL_USERINFO = re.compile(r"(?i)(https?://)[^\s/@]+@")
_URL_QUERY = re.compile(r"(?i)(https?://[^\s?#'\"<>]+)[?#][^\s'\"<>]*")
_TOKEN = re.compile(
    r"\b(?:dapi[a-fA-F0-9]{16,}|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]+|"
    r"sk-[A-Za-z0-9_-]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?)\b"
)


def scrub_failure_text(value: str) -> str:
    """Best-effort credential scrubbing; the result remains sensitive customer content."""
    truncated = len(value) > _MAX_INPUT_CHARS
    value = _ANSI.sub("", value[:_MAX_INPUT_CHARS])
    value = "".join(c for c in value if c in "\n\t" or (c.isprintable() and c != "\x7f"))
    value = _PRIVATE_KEY.sub(_REDACTED, value)
    value = _HEADER.sub(_REDACTED, value)
    value = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    value = _URL_QUERY.sub(r"\1?[REDACTED]", value)
    value = _ASSIGNMENT.sub(_REDACTED, value)
    value = _AUTH_SCHEME.sub(_REDACTED, value)
    value = _TOKEN.sub(_REDACTED, value).strip()
    encoded = value.encode("utf-8", errors="replace")
    if truncated or len(encoded) > MAX_FAILURE_BYTES:
        return (
            encoded[: MAX_FAILURE_BYTES - len(_TRUNCATED)].decode("utf-8", errors="ignore")
            + _TRUNCATED
        )
    return value
