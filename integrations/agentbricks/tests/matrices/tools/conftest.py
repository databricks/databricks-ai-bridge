"""Tools matrix: every test runs once per authoring path."""

from __future__ import annotations

import pytest
from common import AUTHORING_PATHS


@pytest.fixture(params=AUTHORING_PATHS)
def authoring(request: pytest.FixtureRequest) -> str:
    return request.param
