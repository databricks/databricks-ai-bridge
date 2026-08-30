import importlib
import sys

FORBIDDEN_EAGER_IMPORTS = [
    "databricks_openai.mcp_server_toolkit",
    "databricks_openai.vector_search_retriever_tool",
    "unitycatalog.ai.core.base",
    "unitycatalog.ai.core.databricks",
    "unitycatalog.ai.openai.toolkit",
]


def test_package_root_import_does_not_eagerly_import_integrations(monkeypatch):
    monkeypatch.delitem(sys.modules, "databricks_openai", raising=False)
    for module_name in FORBIDDEN_EAGER_IMPORTS:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("databricks_openai")

    loaded = [name for name in FORBIDDEN_EAGER_IMPORTS if name in sys.modules]
    assert not loaded


def test_package_root_client_import_only_loads_client_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "databricks_openai", raising=False)
    monkeypatch.delitem(sys.modules, "databricks_openai.utils.clients", raising=False)
    for module_name in FORBIDDEN_EAGER_IMPORTS:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    databricks_openai = importlib.import_module("databricks_openai")
    client = databricks_openai.DatabricksOpenAI

    assert client.__name__ == "DatabricksOpenAI"
    assert "databricks_openai.utils.clients" in sys.modules

    loaded = [name for name in FORBIDDEN_EAGER_IMPORTS if name in sys.modules]
    assert not loaded
