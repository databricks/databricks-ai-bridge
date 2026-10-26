"""The LangGraph extra and the LangGraph templates pin the same databricks-langchain."""

from pathlib import Path

import pytest
import tomli
from packaging.requirements import Requirement

PACKAGE = Path(__file__).parents[2]
TEMPLATES = PACKAGE / "src/databricks_agentbricks/templates"
PROVIDER = "databricks-langchain"


def _provider(requirements: list[str]) -> list[str]:
    return [
        str(requirement.specifier)
        for requirement in map(Requirement, requirements)
        if requirement.name == PROVIDER
    ]


def _extras() -> dict[str, list[str]]:
    return tomli.loads((PACKAGE / "pyproject.toml").read_text())["project"]["optional-dependencies"]


def test_only_the_langgraph_extra_pins_the_provider():
    pins = {extra: _provider(requirements) for extra, requirements in _extras().items()}
    assert len(pins["langgraph"]) == 1 and pins["langgraph"][0].startswith("==")
    assert pins["openai"] == []


@pytest.mark.parametrize("template", ["agent-langgraph", "custom-agent-langgraph"])
def test_langgraph_templates_pin_the_extras_provider(template):
    dependencies = tomli.loads((TEMPLATES / template / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert _provider(dependencies) == _provider(_extras()["langgraph"])
