"""Shared fixtures for the Agent Bricks CLI unit tests.

Service tests inject their collaborators directly; a directory-wide patch of the old deploy
module would mask the new boundaries and fails because that function no longer exists.
"""

import pytest

from databricks_agentbricks.agent_hint import AGENT_ENV_MARKERS, AGENT_ENV_VALUES


@pytest.fixture(autouse=True)
def human_cli_environment(monkeypatch):
    """Keep ordinary CLI assertions independent of the agent running the tests."""
    for name in (*AGENT_ENV_MARKERS, *AGENT_ENV_VALUES, "TERM_PROGRAM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_AGENT_HINT", raising=False)


@pytest.fixture(autouse=True)
def no_telemetry(monkeypatch):
    """Every command sends telemetry through a real client built from the developer's ambient
    Databricks auth, which shells out and makes network calls; telemetry tests opt back in."""
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "1")
