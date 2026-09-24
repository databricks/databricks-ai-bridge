import importlib

import pytest


def test_agentkit_runtime_exports_are_mason_compatible() -> None:
    import databricks_agentkit
    import databricks_mason

    sdk_names = {
        "AgentKitClient",
        "Memory",
        "MemorySearchResult",
        "MemoryStore",
        "ExtractedMemory",
        "Session",
        "SessionItem",
        "SessionStore",
    }
    runtime_names = set(databricks_mason.__all__) - {
        "MasonClient",
        *sdk_names,
    }

    assert set(databricks_agentkit.__all__) == sdk_names | runtime_names
    for name in sdk_names | runtime_names:
        assert getattr(databricks_agentkit, name) is getattr(databricks_mason, name)


def test_agentkit_runtime_submodules_alias_mason_modules() -> None:
    for module_name in (
        "auth",
        "model_services",
        "store",
        "tool_manifest",
        "workspace",
    ):
        new_module = importlib.import_module(f"databricks_agentkit.runtime.{module_name}")
        old_module = importlib.import_module(f"databricks_mason.runtime.{module_name}")
        assert new_module is old_module

    from databricks_agentkit.runtime.store import InMemoryRuntimeStore
    from databricks_mason.runtime.store import InMemoryRuntimeStore as MasonRuntimeStore

    assert InMemoryRuntimeStore is MasonRuntimeStore


def test_agentkit_runtime_package_imports_workspace_module() -> None:
    from databricks_agentkit.runtime import workspace

    assert workspace is importlib.import_module("databricks_mason.runtime.workspace")


def test_agentkit_framework_packages_are_importable_without_framework_extras() -> None:
    langgraph = importlib.import_module("databricks_agentkit.langgraph")
    openai = importlib.import_module("databricks_agentkit.openai")

    assert langgraph.__all__ == importlib.import_module("databricks_mason.langgraph").__all__
    assert openai.__all__ == importlib.import_module("databricks_mason.openai").__all__


def test_langgraph_invocation_metadata_supports_canonical_and_legacy_keys() -> None:
    pytest.importorskip("langgraph")

    from databricks_agentkit.langgraph.session_store import (
        invocation_id_from_metadata,
        invocation_metadata,
    )

    metadata = invocation_metadata("inv-1")
    assert metadata == {
        "databricks_agentkit.invocation_id": "inv-1",
        "databricks_mason.invocation_id": "inv-1",
    }
    assert invocation_id_from_metadata(metadata) == "inv-1"
    assert invocation_id_from_metadata({"databricks_mason.invocation_id": "legacy"}) == "legacy"
    assert (
        invocation_id_from_metadata(
            {
                "databricks_agentkit.invocation_id": "canonical",
                "databricks_mason.invocation_id": "legacy",
            }
        )
        == "canonical"
    )
    assert invocation_id_from_metadata({}) is None
