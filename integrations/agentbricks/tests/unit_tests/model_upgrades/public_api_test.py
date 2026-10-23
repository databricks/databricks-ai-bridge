"""Smoke test for the public API surface."""


def test_import_public_names():
    from databricks_agentkit.model_upgrades import (
        Result,
        optimize_prompts_and_models,
        promote_to_prod,
        score,
        setup_endpoints,
    )

    assert callable(optimize_prompts_and_models)
    assert callable(score)
    assert callable(promote_to_prod)
    assert callable(setup_endpoints)
    assert isinstance(Result.__dataclass_fields__, dict)


def test_all_exports_match_imports():
    import databricks_agentkit.model_upgrades as model_upgrades

    for name in model_upgrades.__all__:
        assert hasattr(model_upgrades, name), (
            f"__all__ lists '{name}' but module has no such attribute"
        )


def test_no_unexpected_public_names():
    """Public surface stays slim: anything else is internal."""
    import databricks_agentkit.model_upgrades as model_upgrades

    expected = {
        "Result",
        "promote_to_prod",
        "optimize_prompts_and_models",
        "score",
        "setup_endpoints",
    }
    assert set(model_upgrades.__all__) == expected
