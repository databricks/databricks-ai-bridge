"""A newly generated UI must not depend on an unreleased runtime discovery signature."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from databricks_agentbricks.cli.init import _copy_packaged_template
from databricks_agentkit.runtime import model_services


@pytest.mark.parametrize("framework", ["openai", "langgraph"])
def test_scaffold_uses_matching_discovery_with_older_installed_runtime(
    tmp_path, monkeypatch, framework
):
    # This is the public helper signature in older released runtimes. New templates must not
    # silently lose every discovered model by passing its unsupported schema keyword.
    def legacy_discovery(client):
        raise AssertionError("the generated UI must use its matching bundled helper")

    monkeypatch.setattr(model_services, "list_ai_gateway_model_services", legacy_discovery)
    project = tmp_path / "project"
    _copy_packaged_template(f"agent-{framework}", project, (f"ui/agent-{framework}",))
    monkeypatch.syspath_prepend(str(project))
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            monkeypatch.delitem(sys.modules, name)
    try:
        ui = importlib.import_module("runtime.ui")
        assert ui.list_ai_gateway_model_services.__module__ == "runtime.model_services"
        monkeypatch.setattr(ui, "_default_model", lambda: "team.models.default")
        monkeypatch.setattr(ui, "_model_schemas", lambda: ["team.models"])
        calls = []

        def request(method, path, query):
            calls.append(query)
            return {
                "model_services": [
                    {
                        "name": "model-services/team.models.gpt",
                        "supported_api_types": ["openai/v1/chat/completions"],
                    }
                ]
            }

        monkeypatch.setattr(
            ui, "workspace_client", lambda: SimpleNamespace(api_client=SimpleNamespace(do=request))
        )
        assert ui._discover_chat_models()["available"] == ["team.models.default", "team.models.gpt"]
        assert calls == [{"parent": "schemas/team.models", "page_size": 100}]
    finally:
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime."):
                sys.modules.pop(name, None)
