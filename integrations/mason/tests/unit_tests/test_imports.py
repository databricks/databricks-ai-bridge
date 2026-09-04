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
        "Session",
        "SessionItem",
        "SessionStore",
    }
    lazy = {
        "DurableAgentApp",
        "DurableAgentContext",
        "configure_tracing",
        "tag_session",
        "workspace_client",
        "workspace_headers",
    }

    assert set(databricks_mason.__all__) == eager | lazy
    for name in eager:
        assert hasattr(databricks_mason, name)

    from databricks_mason import DurableAgentApp
    from databricks_mason.runtime import DurableAgentApp as RuntimeDurableAgentApp
    from databricks_mason.runtime.app import DurableAgentApp as ModuleDurableAgentApp

    assert DurableAgentApp is RuntimeDurableAgentApp is ModuleDurableAgentApp


def test_durable_runtime_public_surface_is_application_only() -> None:
    import databricks_mason.runtime as runtime

    assert "DurableAgentApp" in runtime.__all__
    assert "DurableAgentContext" in runtime.__all__
    assert "DurableRuntime" not in runtime.__all__
    assert "LakebaseDurabilityStore" not in runtime.__all__
