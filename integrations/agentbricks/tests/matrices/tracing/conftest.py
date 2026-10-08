from __future__ import annotations

import pathlib

import pytest


@pytest.fixture(scope="session")
def scratch_schema(deploy_scratch_schema) -> str:
    return deploy_scratch_schema(pathlib.Path(__file__).parent / "bundle")
