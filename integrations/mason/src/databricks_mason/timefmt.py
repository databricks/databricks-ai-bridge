"""Timestamp parsing and humanization for the Mason CLI.

The agents/v1 APIs return timestamps two ways: memory *stores* use epoch-millis
int64 (`created_at`/`updated_at`), while entries, sessions, and session items use
`google.protobuf.Timestamp`, which serializes to an RFC 3339 string
(`2026-08-15T01:29:00Z`). `parse_timestamp` accepts either.

`relative` returns concise phrases such as "13 days ago" and "An hour ago".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union

TimestampValue = Union[int, float, str, datetime, None]


def parse_timestamp(value: TimestampValue) -> Optional[datetime]:
    """Parse a datetime, an epoch-millis number, a numeric string, or an RFC 3339 string.

    Returns a timezone-aware UTC datetime, or None if the value is empty/unparseable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
    if isinstance(value, str):
        if value.isdigit():
            return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
        iso = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def relative(value: TimestampValue, *, now: Optional[datetime] = None) -> str:
    """Humanize a timestamp as an "N units ago" string."""
    dt = parse_timestamp(value)
    if dt is None:
        return "—"
    now = now or datetime.now(tz=timezone.utc)
    seconds = (now - dt).total_seconds()
    if seconds < 0:
        return "just now"

    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    if seconds < 45:
        return "just now"
    if minutes < 2:
        return "A minute ago"
    if minutes < 60:
        return f"{round(minutes)} minutes ago"
    if hours < 2:
        return "An hour ago"
    if hours < 24:
        return f"{round(hours)} hours ago"
    if days < 2:
        return "A day ago"
    if days < 30:
        return f"{round(days)} days ago"
    if days < 60:
        return "A month ago"
    if days < 365:
        return f"{round(days / 30)} months ago"
    if days < 730:
        return "A year ago"
    return f"{round(days / 365)} years ago"


def absolute(value: TimestampValue) -> str:
    """Format a timestamp like the mock's detail rail: "May 19, 2026, 05:00 PM"."""
    dt = parse_timestamp(value)
    if dt is None:
        return "—"
    local = dt.astimezone()
    return f"{local.strftime('%b')} {local.day}, {local.strftime('%Y, %I:%M %p')}"
