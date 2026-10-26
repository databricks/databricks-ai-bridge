"""Shared pagination validation for typed Agent Bricks resources."""

from typing import Optional

_MAX_PAGE_SIZE = 100


def _validate_bound(value: Optional[int], name: str, maximum: int = _MAX_PAGE_SIZE) -> None:
    if value is None:
        return
    # Reject non-integers (and bool, which is an int subclass) up front: without this a str
    # raised a cryptic ``'<=' not supported between 'int' and 'str'`` TypeError, and a bool/float
    # slipped through the range check (True == 1, 1.5 in [1, 100]) to reach the API as garbage.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def validate_page_size(page_size: Optional[int], *, maximum: int = _MAX_PAGE_SIZE) -> None:
    _validate_bound(page_size, "page_size", maximum)


def validate_limit(limit: Optional[int]) -> None:
    _validate_bound(limit, "limit")
