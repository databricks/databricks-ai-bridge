"""A published CLI deploys the paired SDK without requiring repository sources."""

from unittest.mock import Mock

import pytest

from databricks_agentbricks.cli import prepared_deploy
from databricks_agentbricks.errors import AgentCliError


@pytest.fixture
def installed_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepared_deploy,
        "__file__",
        str(tmp_path / "site-packages/databricks_agentbricks/cli/prepared_deploy.py"),
    )
    build = Mock(side_effect=AssertionError("Installed packages must not build local source"))
    monkeypatch.setattr(prepared_deploy.subprocess, "run", build)
    return tmp_path, build


def test_published_install_stages_the_exact_sdk_without_cli(installed_cli, monkeypatch):
    target, build = installed_cli
    monkeypatch.setattr(prepared_deploy, "version", lambda package: "0.11.0")
    prepared_deploy._stage_runtime_requirements(target)
    assert (target / "requirements.txt").read_text() == (
        "databricks-agentkit-runtime[sandbox]==0.11.0\n"
    )
    assert not list(target.glob("*.whl"))
    build.assert_not_called()


def test_mismatched_install_fails_before_writing_a_deployment(installed_cli, monkeypatch):
    target, _ = installed_cli
    versions = {"databricks-agentbricks": "0.11.0", "databricks-agentkit-runtime": "0.10.0"}
    monkeypatch.setattr(prepared_deploy, "version", versions.__getitem__)
    with pytest.raises(AgentCliError, match="versions must match"):
        prepared_deploy._stage_runtime_requirements(target)
    assert not (target / "requirements.txt").exists()


def test_suite_pins_both_wheels_for_transitive_sdk_resolution(tmp_path):
    import tomli
    from harness.project import _pin_project_wheels

    wheels = tmp_path / "dist"
    wheels.mkdir()
    cli = wheels / "databricks_agentbricks-0.11.0-py3-none-any.whl"
    sdk = wheels / "databricks_agentkit_runtime-0.11.0-py3-none-any.whl"
    cli.write_bytes(b"cli")
    sdk.write_bytes(b"sdk")
    project = tmp_path / "agent"
    project.mkdir()
    metadata = project / "pyproject.toml"
    metadata.write_text('[project]\ndependencies=["databricks-agentbricks[openai]"]\n')
    _pin_project_wheels(project, cli, None, sdk)
    document = tomli.loads(metadata.read_text())
    assert "databricks-agentkit-runtime" in document["project"]["dependencies"]
    sources = document["tool"]["uv"]["sources"]
    assert (project / sources["databricks-agentbricks"]["path"]).read_bytes() == b"cli"
    assert (project / sources["databricks-agentkit-runtime"]["path"]).read_bytes() == b"sdk"
