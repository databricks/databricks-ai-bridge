"""Shared pagination validation for typed Mason resources."""

from typing import Optional

_MAX_PAGE_SIZE = 100


def validate_page_size(page_size: Optional[int]) -> None:
    if page_size is not None and not 1 <= page_size <= _MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")
