def test_package_import() -> None:
    import databricks_mason

    assert databricks_mason.__doc__


def test_public_surface() -> None:
    import databricks_mason

    for name in ("MasonClient", "AgentCliError", "memory_store_path", "memory_entry_path"):
        assert name in databricks_mason.__all__
        assert hasattr(databricks_mason, name)


def test_agentapiclient_alias() -> None:
    from databricks_mason import MasonClient
    from databricks_mason.client import AgentApiClient

    assert AgentApiClient is MasonClient
