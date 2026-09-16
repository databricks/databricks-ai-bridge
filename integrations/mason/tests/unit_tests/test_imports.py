def test_package_import() -> None:
    import databricks_mason

    assert databricks_mason.__doc__


def test_public_surface() -> None:
    import databricks_mason

    eager = {
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
        "AgentApp",
        "InvocationContext",
        "configure_tracing",
        "start_trace",
        "workspace_client",
        "workspace_headers",
    }

    assert set(databricks_mason.__all__) == eager | lazy
    for name in eager:
        assert hasattr(databricks_mason, name)

    from databricks_mason import AgentApp
    from databricks_mason.runtime import AgentApp as RuntimeAgentApp

    assert AgentApp is RuntimeAgentApp


def test_runtime_public_surface_is_application_only() -> None:
    import databricks_mason.runtime as runtime

    assert "AgentApp" in runtime.__all__
    assert "InvocationContext" in runtime.__all__
    assert "Runtime" not in runtime.__all__
    assert "LakebaseDurableRuntimeStore" not in runtime.__all__
