"""Shared pagination validation for typed Agent Bricks resources."""

from typing import Optional

_MAX_PAGE_SIZE = 100


def _validate_bound(value: Optional[int], name: str) -> None:
    if value is None:
        return
    # Reject non-integers (and bool, which is an int subclass) up front: without this a str
    # raised a cryptic ``'<=' not supported between 'int' and 'str'`` TypeError, and a bool/float
    # slipped through the range check (True == 1, 1.5 in [1, 100]) to reach the API as garbage.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between 1 and {_MAX_PAGE_SIZE}")
    if not 1 <= value <= _MAX_PAGE_SIZE:
        raise ValueError(f"{name} must be between 1 and {_MAX_PAGE_SIZE}")


def validate_page_size(page_size: Optional[int]) -> None:
    _validate_bound(page_size, "page_size")


def validate_limit(limit: Optional[int]) -> None:
    _validate_bound(limit, "limit")
