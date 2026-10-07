"""Tests for the model-service CRUD helpers in `databricks_agentkit.runtime.model_services`.

These are the one place the UC model-services request shapes live (the CLI's API client and the
model-upgrade optimizer both build on them), so the shapes are pinned here.
"""

from __future__ import annotations

import pytest

from databricks_agentkit.runtime import model_services as ms

SERVICE = "main.my_agent.llm"
SYSTEM_SONNET = "/api/2.1/unity-catalog/model-services/system.ai.claude-sonnet-4-5"


@pytest.fixture(autouse=True)
def _fresh_foundation_cache(monkeypatch):
    monkeypatch.setattr(ms, "_foundation_cache", {})


class _Api:
    def __init__(self, responses=None):
        self.calls: list[tuple] = []
        self._responses = responses or {}

    def do(self, method, path, query=None, body=None):
        self.calls.append((method, path, query, body))
        return self._responses.get((method, path), {})


class _Schemas:
    def __init__(self, error=None):
        self.created: list[tuple[str, str]] = []
        self._error = error

    def create(self, name, catalog_name):
        self.created.append((catalog_name, name))
        if self._error is not None:
            raise self._error


class _Client:
    def __init__(self, responses=None, schema_error=None):
        self.api_client = _Api(responses)
        self.schemas = _Schemas(schema_error)


def _service(model):
    """A model service routed to ``model`` (a model-service name), as UC returns it."""
    return {
        "config": {
            "routing": {"destinations": [ms.ppt_destination(ms.foundation_model(None, model))]}
        }
    }


def test_names():
    assert ms.system_ai_name("claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    assert ms.system_ai_name("system.ai.claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    # An underscore, not a hyphen: the clone's name is a UC identifier.
    assert ms.exp_name(SERVICE) == "main.my_agent.llm_exp"
    assert ms.delete_command(SERVICE) == (
        "databricks api delete /api/2.1/unity-catalog/model-services/main.my_agent.llm"
    )


def test_ppt_destination_shape():
    assert ms.ppt_destination("system.ai.databricks-claude-haiku-4-5") == {
        "name": "primary",
        "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
        "pay_per_token_config": {"model": "models/system.ai.databricks-claude-haiku-4-5"},
        "traffic_percentage": 100,
    }


def test_foundation_model_reads_the_system_services_destination():
    # Agents call system.ai.claude-sonnet-4-5; its destination is the registered model behind it.
    routed = ms.ppt_destination("system.ai.databricks-claude-sonnet-4-5")
    client = _Client({("GET", SYSTEM_SONNET): {"config": {"routing": {"destinations": [routed]}}}})
    for name in ("claude-sonnet-4-5", "system.ai.claude-sonnet-4-5"):
        assert ms.foundation_model(client, name) == "system.ai.databricks-claude-sonnet-4-5"
    assert len(client.api_client.calls) == 1  # cached


def test_foundation_model_falls_back_to_the_naming_convention():
    assert ms.foundation_model(None, "gpt-5-4-mini") == "system.ai.databricks-gpt-5-4-mini"
    assert ms.foundation_model(None, "system.ai.databricks-gpt-5-4-mini") == (
        "system.ai.databricks-gpt-5-4-mini"
    )


def test_destination_model_maps_back_to_the_model_service_name():
    assert ms.destination_model(_service("claude-haiku-4-5")) == "system.ai.claude-haiku-4-5"
    assert ms.destination_model({"config": {}}) is None


def test_create_makes_schema_then_posts_under_parent():
    client = _Client()
    ms.create(client, SERVICE, "claude-sonnet-4-5", comment="managed_by=x")
    assert client.schemas.created == [("main", "my_agent")]
    (method, path, query, body) = client.api_client.calls[-1]
    assert (method, path) == ("POST", "/api/2.1/unity-catalog/model-services")
    assert query == {"parent": "schemas/main.my_agent", "model_service_id": "llm"}
    destination = body["config"]["routing"]["destinations"][0]
    assert destination["pay_per_token_config"] == {
        "model": "models/system.ai.databricks-claude-sonnet-4-5"
    }
    assert body["comment"] == "managed_by=x"


@pytest.mark.parametrize(
    "error",
    [Exception("Schema 'my_agent' already exists"), Exception("ALREADY_EXISTS: already exists")],
)
def test_create_tolerates_existing_schema(error):
    client = _Client(schema_error=error)
    ms.create(client, SERVICE, "claude-sonnet-4-5")
    assert client.api_client.calls[-1][0] == "POST"


def test_create_surfaces_real_schema_errors():
    client = _Client(schema_error=PermissionError("no USE CATALOG on main"))
    with pytest.raises(PermissionError):
        ms.create(client, SERVICE, "claude-sonnet-4-5")
    assert not [c for c in client.api_client.calls if c[0] == "POST"]


def test_get_set_delete():
    path = "/api/2.1/unity-catalog/model-services/main.my_agent.llm"
    client = _Client({("GET", path): _service("claude-sonnet-4-5")})
    assert ms.get_model(client, SERVICE) == "system.ai.claude-sonnet-4-5"

    ms.set_model(client, SERVICE, "claude-haiku-4-5")
    method, set_path, query, body = client.api_client.calls[-1]
    assert (method, set_path) == ("PATCH", path)
    assert query == {"update_mask": "config.routing.destinations"}
    assert body["config"]["routing"]["destinations"][0]["pay_per_token_config"] == {
        "model": "models/system.ai.databricks-claude-haiku-4-5"
    }

    ms.delete(client, SERVICE)
    assert client.api_client.calls[-1][:2] == ("DELETE", path)


def test_get_model_without_destination_raises():
    path = "/api/2.1/unity-catalog/model-services/main.my_agent.llm"
    client = _Client({("GET", path): {"config": {}}})
    with pytest.raises(ValueError, match="no foundation-model destination"):
        ms.get_model(client, SERVICE)


def test_grant_requests_execute_required_parents_best_effort():
    requests = ms.grant_requests(SERVICE, "sp-app-id")
    assert [(path, required) for path, _, required in requests] == [
        ("/api/2.1/unity-catalog/permissions/catalog/main", False),
        ("/api/2.1/unity-catalog/permissions/schema/main.my_agent", False),
        ("/api/2.1/unity-catalog/permissions/model_service/main.my_agent.llm", True),
    ]
    assert requests[-1][1] == {"changes": [{"principal": "sp-app-id", "add": ["EXECUTE"]}]}


def test_can_execute_reads_effective_permissions():
    perms = "/api/2.1/unity-catalog/effective-permissions/function/system.ai.databricks-gpt-5"
    granted = {"privilege_assignments": [{"privileges": [{"privilege": "EXECUTE"}]}]}
    manage_only = {"privilege_assignments": [{"privileges": [{"privilege": "MANAGE"}]}]}
    assert ms.can_execute(_Client({("GET", perms): granted}), "gpt-5", "me") is True
    assert ms.can_execute(_Client({("GET", perms): manage_only}), "gpt-5", "me") is False


def test_can_execute_is_unknown_when_grants_are_unreadable():
    class _Denied(_Api):
        def do(self, method, path, query=None, body=None):
            if "effective-permissions" in path:
                raise PermissionError("no")
            return super().do(method, path, query, body)

    client = _Client()
    client.api_client = _Denied()
    assert ms.can_execute(client, "gpt-5", "me") is None


def test_prompt_access_grants_the_schema_privileges_the_registry_requires():
    from databricks_agentkit.runtime import prompt_registry

    requests = prompt_registry.grant_requests("main.my_agent", "sp-1")
    (catalog_path, catalog_body, catalog_required), (schema_path, schema_body, required) = requests
    assert catalog_path.endswith("/catalog/main") and catalog_required is False
    assert schema_path.endswith("/schema/main.my_agent") and required is True
    privileges = schema_body["changes"][0]["add"]
    assert set(privileges) == {"USE_SCHEMA", "EXECUTE", "CREATE_FUNCTION", "MANAGE"}
    assert prompt_registry.schema_of("main.my_agent.writer") == "main.my_agent"
