"""Unit tests for the shared pagination bound validators."""

from __future__ import annotations

import pytest

from databricks_mason._pagination import validate_limit, validate_page_size


@pytest.mark.parametrize("fn,name", [(validate_page_size, "page_size"), (validate_limit, "limit")])
class TestValidateBound:
    def test_none_is_allowed(self, fn, name):
        fn(None)  # no exception

    @pytest.mark.parametrize("value", [1, 50, 100])
    def test_in_range_ok(self, fn, name, value):
        fn(value)  # no exception

    @pytest.mark.parametrize("value", [0, -1, 101, 10**9])
    def test_out_of_range_raises_with_name(self, fn, name, value):
        with pytest.raises(ValueError, match=f"{name} must be between 1 and 100"):
            fn(value)

    @pytest.mark.parametrize("value", ["10", 1.5, [1], object()])
    def test_non_integer_raises_clearly_not_typeerror(self, fn, name, value):
        # Regression: a str previously raised a cryptic TypeError from the comparison.
        with pytest.raises(ValueError, match=f"{name} must be an integer"):
            fn(value)

    def test_bool_is_rejected(self, fn, name):
        # bool is an int subclass; True==1 would otherwise slip through as page_size=1.
        with pytest.raises(ValueError, match=f"{name} must be an integer"):
            fn(True)
