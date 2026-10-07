"""Tests for prompt-URI parsing and required-var extraction."""

import pytest

from databricks_agentkit.model_upgrades.optimization import (
    _extract_required_vars,
    _parse_prompt_uri,
)


def test_parse_alias_uri():
    assert _parse_prompt_uri("prompts:/cat.schema.supervisor@production") == (
        "cat.schema.supervisor",
        "production",
        None,
        "supervisor",
    )


def test_parse_version_uri():
    assert _parse_prompt_uri("prompts:/cat.schema.foo/3") == (
        "cat.schema.foo",
        None,
        "3",
        "foo",
    )


def test_parse_no_alias_defaults_to_production():
    assert _parse_prompt_uri("prompts:/cat.schema.foo") == (
        "cat.schema.foo",
        "production",
        None,
        "foo",
    )


def test_parse_invalid_uri_raises():
    with pytest.raises(ValueError, match="Invalid prompt URI"):
        _parse_prompt_uri("not-a-uri")


def test_extract_jinja_vars():
    assert _extract_required_vars("Hello {{ name }}, {{ greeting }}!") == ["name", "greeting"]


def test_extract_python_vars():
    assert _extract_required_vars("Hello {name}, {greeting}!") == ["name", "greeting"]


def test_extract_dedupes():
    assert _extract_required_vars("{{ x }} and {{ x }} and {x}") == ["x"]


def test_extract_empty():
    assert _extract_required_vars("no placeholders here") == []
