"""Generated projects use the bridge revision selected by the runner."""

import pathlib
import runpy

import pytest
import tomli

_MATRIX = pathlib.Path(__file__).resolve().parents[1] / "e2e" / "tool_matrix.py"
_NAMESPACE = runpy.run_path(str(_MATRIX))
Runner = _NAMESPACE["Runner"]
MatrixError = _NAMESPACE["MatrixError"]
BRIDGE_SHA = "a" * 40


@pytest.mark.parametrize("authoring", ["cli", "direct"])
def test_generated_project_pins_app_packages_to_bridge_sha(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, authoring: str
) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")
    runner = Runner(None, tmp_path, wheel, bridge_sha=BRIDGE_SHA)
    runner.uc_function = "main.test_schema.marker"
    monkeypatch.setitem(runner.create_projects.__globals__, "AUTHORING_PATHS", (authoring,))

    def scaffold(_label: str, _args: list[str], **_kwargs: object) -> str:
        project = tmp_path / "projects" / f"langgraph-{authoring}"
        project.mkdir(parents=True)
        langchain_requirement = ', "databricks-langchain>=0.17.0"' if authoring == "direct" else ""
        project.joinpath("pyproject.toml").write_text(
            '[project]\nname = "test-agent"\ndependencies = '
            f'["databricks-agentbricks[langgraph]>=0.2.0"{langchain_requirement}]\n'
        )
        return ""

    monkeypatch.setattr(runner, "run_long", scaffold)
    monkeypatch.setattr(runner, "_author_cli", lambda _project: None)
    monkeypatch.setattr(runner, "_author_direct", lambda _project, _framework: None)

    cases = runner.create_projects()

    assert len(cases) == 1
    pyproject = tomli.loads((cases[0].path / "pyproject.toml").read_text())
    assert "databricks-langchain>=0.17.0" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["dependencies"].count("databricks-langchain>=0.17.0") == 1
    assert pyproject["tool"]["uv"]["sources"]["databricks-langchain"] == {
        "git": "https://github.com/databricks/databricks-ai-bridge.git",
        "rev": BRIDGE_SHA,
        "subdirectory": "integrations/langchain",
    }
    assert pyproject["tool"]["uv"]["sources"]["databricks-agentbricks"] == {
        "git": "https://github.com/databricks/databricks-ai-bridge.git",
        "rev": BRIDGE_SHA,
        "subdirectory": "integrations/agentbricks",
    }


def test_bridge_source_requires_immutable_sha(tmp_path: pathlib.Path) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"test wheel")

    with pytest.raises(MatrixError, match="40-character"):
        Runner(None, tmp_path, wheel, bridge_sha="main")
