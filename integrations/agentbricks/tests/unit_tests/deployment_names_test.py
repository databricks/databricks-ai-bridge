"""The shared deployment-name budget and suggestion in ``services/deployment/names.py``."""

from __future__ import annotations

import hashlib

import pytest

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.names import (
    MAX_BASE_NAME_LEN,
    DeploymentName,
    _prefixed_name,
    base_name_problem,
    fit_base_name,
)


def test_budget_leaves_17_characters_after_the_prefix():
    assert MAX_BASE_NAME_LEN == 17


@pytest.mark.parametrize("name", ["my-agent", "a" * 17, "agent-2"])
def test_fitting_names_are_unchanged(name: str):
    assert fit_base_name(name) == name
    assert base_name_problem(name) is None


def test_long_name_keeps_ten_characters_and_a_hash_token():
    fitted = fit_base_name("caj-langgraph-agent")
    token = hashlib.sha256(b"caj-langgraph-agent").hexdigest()[:6]

    assert fitted == f"caj-langgr-{token}"
    assert len(fitted) <= MAX_BASE_NAME_LEN
    # The hash keeps suggestions stable across runs.
    assert fit_base_name("caj-langgraph-agent") == fitted


def test_long_name_uses_the_given_token():
    assert fit_base_name("caj-langgraph-agent", "ilmmyx") == "caj-langgr-ilmmyx"


def test_long_name_strips_trailing_hyphens_before_the_token():
    assert fit_base_name("abcdefghi--xyz-more-words", "tokenx") == "abcdefghi-tokenx"


@pytest.mark.parametrize(
    ("raw", "fitted"),
    [
        ("My_Agent", "my-agent"),
        ("my agent", "my-agent"),
        ("MY.AGENT!", "my-agent"),
        ("--edge--", "edge"),
        ("agent-bricks-my-agent", "my-agent"),
    ],
)
def test_names_are_sanitized(raw: str, fitted: str):
    assert fit_base_name(raw) == fitted


@pytest.mark.parametrize("raw", ["", "___", "!!!", "agent-bricks-"])
def test_names_that_sanitize_to_nothing_fall_back_to_agent(raw: str):
    assert fit_base_name(raw) == "agent"


@pytest.mark.parametrize(
    "raw", ["caj-langgraph-agent", "prenick-dev-openai-agent", "Org_Team_Env_Project_Iterator"]
)
def test_fitted_names_always_pass_deploy_validation(raw: str):
    fitted = fit_base_name(raw)

    assert base_name_problem(fitted) is None
    assert DeploymentName(_prefixed_name(fitted)) == f"agent-bricks-{fitted}"


def test_base_name_problem_reports_length_and_charset():
    assert base_name_problem("a" * 18) == (
        f"'{'a' * 18}' is too long to deploy (agent-bricks-{'a' * 18} is 31 > 30)"
    )
    assert "is not a valid Databricks Apps name" in (base_name_problem("My_Agent") or "")
    assert "is not a valid Databricks Apps name" in (base_name_problem("") or "")
    # A name that already carries the prefix is judged as the full app name.
    assert base_name_problem("agent-bricks-my-agent") is None


def test_deployment_name_length_hint_states_the_budget():
    with pytest.raises(AgentCliError, match="too long") as excinfo:
        DeploymentName("agent-bricks-caj-langgraph-agent")

    assert excinfo.value.hint is not None
    assert "Names can be up to 17 characters" in excinfo.value.hint
    assert f"'agent-bricks-{fit_base_name('caj-langgraph-agent')}'" in excinfo.value.hint
