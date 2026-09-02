from __future__ import annotations

import ast

import pytest

from databricks_mason.errors import AgentCliError
from databricks_mason.integration_codegen import (
    IntegrationRegistry,
    registry_relative_path,
    render_registry,
)
from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction


def test_registry_round_trips_canonical_python_without_executing_it(tmp_path) -> None:
    registry = IntegrationRegistry.empty(tmp_path)
    registry.add(
        Sandbox(
            id="sandbox",
            scopes=(Scope.table("samples.nyctaxi.trips"),),
        )
    )
    registry.add(MCPService(id="web", service="system.ai.web_search"))
    registry.add(UCFunction(id="lookup", function="main.tools.lookup"))

    path = registry.write()
    source = path.read_text(encoding="utf-8")

    ast.parse(source)
    assert "DATABRICKS_TOOLS" in source
    assert IntegrationRegistry.load(tmp_path).integrations == registry.integrations
    assert registry.definition_line("sandbox") > 0


def test_registry_rejects_dynamic_python_without_running_it(tmp_path) -> None:
    marker = tmp_path / "executed"
    path = tmp_path / "agent" / "databricks_tools.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('bad')\n"
        "DATABRICKS_TOOLS = []\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentCliError, match="CLI-owned|canonical"):
        IntegrationRegistry.load(tmp_path)

    assert not marker.exists()


def test_registry_add_is_idempotent_and_conflicts_by_id(tmp_path) -> None:
    registry = IntegrationRegistry.empty(tmp_path)
    sandbox = Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))

    assert registry.add(sandbox) is True
    assert registry.add(sandbox) is False

    with pytest.raises(AgentCliError, match="different configuration"):
        registry.add(
            Sandbox(
                id="sandbox",
                scopes=(Scope.volume("main.data.other"),),
            )
        )


def test_empty_registry_is_a_valid_importable_no_op(tmp_path) -> None:
    registry = IntegrationRegistry.empty(tmp_path)

    path = registry.write()

    assert IntegrationRegistry.load(tmp_path).integrations == []
    assert "DATABRICKS_TOOLS" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.replace(
            '        id="web",',
            '        id="web",\n        id="duplicate",',
        ),
        lambda source: source.replace(
            "DATABRICKS_TOOLS: tuple[mason_integrations.Integration, ...]",
            "DATABRICKS_TOOLS: object",
        ),
    ],
)
def test_registry_rejects_python_that_is_not_compileable_canonical_source(tmp_path, mutate) -> None:
    path = tmp_path / "agent" / "databricks_tools.py"
    path.parent.mkdir(parents=True)
    canonical = render_registry([MCPService(id="web", service="system.ai.web_search")])
    path.write_text(mutate(canonical), encoding="utf-8")

    with pytest.raises(AgentCliError, match="canonical|parse"):
        IntegrationRegistry.load(tmp_path)


def test_registry_wraps_failure_to_create_its_parent_directory(tmp_path) -> None:
    (tmp_path / "agent").write_text("not a directory", encoding="utf-8")

    with pytest.raises(AgentCliError, match="Could not write"):
        IntegrationRegistry.empty(tmp_path).write()


def test_registry_uses_the_supported_framework_package_path(tmp_path) -> None:
    relative_path = registry_relative_path("openai")
    registry = IntegrationRegistry.empty(tmp_path, relative_path=relative_path)

    path = registry.write()

    assert path == tmp_path / "agent" / "databricks_tools.py"
    assert (
        IntegrationRegistry.load(
            tmp_path,
            relative_path=relative_path,
        ).integrations
        == []
    )
