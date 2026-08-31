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
        "configure_tracing",
        "tag_session",
        "workspace_client",
        "workspace_headers",
    }

    assert set(databricks_mason.__all__) == eager | lazy
    for name in eager:
        assert hasattr(databricks_mason, name)
