"""Shared fixtures for the Agent Bricks CLI unit tests.

Service tests inject their collaborators directly; a directory-wide patch of the old deploy
module would mask the new boundaries and fails because that function no longer exists.
"""

import pytest


@pytest.fixture(autouse=True)
def suppress_cli_hints(monkeypatch):
    """Isolate command-output assertions; hint tests exercise the default separately."""
    monkeypatch.setenv("AGENTBRICKS_DISABLE_AGENT_HINT", "1")
