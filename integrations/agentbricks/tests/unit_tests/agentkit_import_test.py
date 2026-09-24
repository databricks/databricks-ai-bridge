import importlib

import pytest


def test_agentkit_runtime_submodules_alias_implementation_modules() -> None:
    for module_name in (
        "auth",
        "model_services",
        "store",
        "tool_manifest",
        "workspace",
    ):
        public_module = importlib.import_module(f"databricks_agentkit.runtime.{module_name}")
        implementation_module = importlib.import_module(
            f"databricks_agentbricks.runtime.{module_name}"
        )
        assert public_module is implementation_module

    from databricks_agentbricks.runtime.store import InMemoryRuntimeStore as ImplementationStore
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    assert InMemoryRuntimeStore is ImplementationStore


def test_agentkit_runtime_package_imports_workspace_module() -> None:
    from databricks_agentkit.runtime import workspace

    assert workspace is importlib.import_module("databricks_agentbricks.runtime.workspace")


def test_agentkit_framework_packages_are_importable_without_framework_extras() -> None:
    langgraph = importlib.import_module("databricks_agentkit.langgraph")
    openai = importlib.import_module("databricks_agentkit.openai")

    assert langgraph.__all__ == importlib.import_module("databricks_agentbricks.langgraph").__all__
    assert openai.__all__ == importlib.import_module("databricks_agentbricks.openai").__all__


def test_langgraph_invocation_metadata_uses_the_agentkit_key() -> None:
    pytest.importorskip("langgraph")

    from databricks_agentkit.langgraph.session_store import (
        invocation_id_from_metadata,
        invocation_metadata,
    )

    metadata = invocation_metadata("inv-1")
    assert metadata == {"databricks_agentkit.invocation_id": "inv-1"}
    assert invocation_id_from_metadata(metadata) == "inv-1"
    assert invocation_id_from_metadata({"databricks_other.invocation_id": "unknown"}) is None
    assert invocation_id_from_metadata({}) is None
