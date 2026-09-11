"""Unit tests for the databricks-mason source pin helpers (read/write + pin shapes)."""

from __future__ import annotations

import pathlib

import tomli

from databricks_mason import mason_source


def test_git_builds_pin_and_converts_local_path_to_file_uri(tmp_path: pathlib.Path):
    assert mason_source.git("https://github.com/x/y.git", "abc") == {
        "git": "https://github.com/x/y.git",
        "rev": "abc",
        "subdirectory": "integrations/mason",
    }
    # A bare local path (and a git+ prefix) becomes a file:// URL uv can clone.
    local = mason_source.git(f"git+{tmp_path}", "abc")
    assert local["git"] == tmp_path.resolve().as_uri()


def test_editable_pin():
    assert mason_source.editable(pathlib.Path("/repo/integrations/mason")) == {
        "path": "/repo/integrations/mason",
        "editable": True,
    }


def test_read_returns_pin_or_none(tmp_path: pathlib.Path):
    pyproject = tmp_path / "pyproject.toml"
    assert mason_source.read(pyproject) is None  # missing file
    pyproject.write_text('[project]\nname = "t"\n')
    assert mason_source.read(pyproject) is None  # no pin
    pyproject.write_text(
        '[project]\nname = "t"\n\n[tool.uv.sources]\ndatabricks-mason = { path = "x" }\n'
    )
    assert mason_source.read(pyproject) == {"path": "x"}


def test_write_sets_pin_and_preserves_existing_content(tmp_path: pathlib.Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "t"\ndependencies = ["databricks-mason>=0.1"]\n')

    mason_source.write(pyproject, mason_source.git("https://github.com/x/y.git", "abc"))

    data = tomli.loads(pyproject.read_text())
    assert data["project"]["dependencies"] == ["databricks-mason>=0.1"]  # untouched
    assert data["tool"]["uv"]["sources"]["databricks-mason"] == {
        "git": "https://github.com/x/y.git",
        "rev": "abc",
        "subdirectory": "integrations/mason",
    }
