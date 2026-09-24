def test_package_import() -> None:
    import databricks_mason

    assert databricks_mason.__doc__


def test_public_surface() -> None:
    import databricks_mason

    eager = {
        "AgentKitClient",
        "MasonClient",
        "Memory",
        "MemorySearchResult",
        "MemoryStore",
        "ExtractedMemory",
        "Session",
        "SessionItem",
        "SessionStore",
    }
    lazy = {
        "DurableAgentServer",
        "AgentApp",
        "InvocationContext",
        "configure_tracing",
        "list_ai_gateway_model_services",
        "start_trace",
        "workspace_client",
        "workspace_headers",
    }

    assert set(databricks_mason.__all__) == eager | lazy
    for name in eager:
        assert hasattr(databricks_mason, name)

    from databricks_mason import AgentApp, DurableAgentServer
    from databricks_mason.runtime import AgentApp as RuntimeAgentApp
    from databricks_mason.runtime import DurableAgentServer as RuntimeDurableAgentServer

    assert DurableAgentServer is RuntimeDurableAgentServer
    assert AgentApp is DurableAgentServer
    assert RuntimeAgentApp is DurableAgentServer


def test_agentkit_import_uses_public_client() -> None:
    import databricks_agentkit
    import databricks_mason

    public_sdk_types = {
        "AgentKitClient",
        "Memory",
        "MemorySearchResult",
        "MemoryStore",
        "ExtractedMemory",
        "Session",
        "SessionItem",
        "SessionStore",
    }
    runtime_helpers = {
        "DurableAgentServer",
        "AgentApp",
        "InvocationContext",
        "configure_tracing",
        "list_ai_gateway_model_services",
        "start_trace",
        "workspace_client",
        "workspace_headers",
    }

    assert set(databricks_agentkit.__all__) == public_sdk_types | runtime_helpers
    for name in public_sdk_types | runtime_helpers:
        assert getattr(databricks_agentkit, name) is getattr(databricks_mason, name)


def test_runtime_public_surface_is_application_only() -> None:
    import databricks_mason.runtime as runtime

    assert "DurableAgentServer" in runtime.__all__
    assert "AgentApp" in runtime.__all__
    assert "InvocationContext" in runtime.__all__
    assert "Runtime" not in runtime.__all__
    assert "LakebaseDurableRuntimeStore" not in runtime.__all__


def test_agent_app_compatibility_alias_constructs_server() -> None:
    from databricks_mason import AgentApp, DurableAgentServer
    from databricks_mason.runtime.store import InMemoryRuntimeStore

    server = AgentApp(runtime_store=InMemoryRuntimeStore())

    assert type(server) is DurableAgentServer
