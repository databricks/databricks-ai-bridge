"""Starter test for the Mason-scaffolded Python tool."""

import pytest
from agent.tools.__MASON_TOOL_MODULE__ import (  # ty: ignore[unresolved-import]
    __MASON_TOOL_FUNCTION__,
)

pytestmark = pytest.mark.skip(reason="Implement __MASON_TOOL_FUNCTION__ and enable this test.")


def test___MASON_TOOL_FUNCTION___returns_a_result():
    assert __MASON_TOOL_FUNCTION__("example") is not None
