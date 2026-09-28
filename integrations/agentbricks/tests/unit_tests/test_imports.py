def test_public_package_import() -> None:
    import databricks_agentkit

    assert databricks_agentkit.__doc__


def test_public_surface() -> None:
    import databricks_agentkit

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
        assert hasattr(databricks_agentkit, name)


def test_runtime_public_surface_is_application_only() -> None:
    import databricks_agentkit.runtime as runtime

    assert "DurableAgentServer" in runtime.__all__
    assert "AgentApp" in runtime.__all__
    assert "InvocationContext" in runtime.__all__
    assert "Runtime" not in runtime.__all__
    assert "LakebaseDurableRuntimeStore" not in runtime.__all__


def test_agent_app_compatibility_alias_constructs_server() -> None:
    from databricks_agentkit import AgentApp, DurableAgentServer
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    server = AgentApp(runtime_store=InMemoryRuntimeStore())

    assert type(server) is DurableAgentServer
