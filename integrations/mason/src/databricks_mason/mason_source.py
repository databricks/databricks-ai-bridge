"""Read and write the databricks-mason source pin in a generated project's pyproject.toml.

A Mason project can override where `databricks-mason` is installed from via `[tool.uv.sources]`.
`mason init` writes that pin (a git repo or a local editable checkout) and `mason deploy` reads it
to reject a machine-local pin the Apps build can't reach. This module owns the pin's shapes and the
tomlkit plumbing so init and deploy don't each reimplement them.
"""

from __future__ import annotations

import pathlib

import tomli
import tomlkit

_SUBDIRECTORY = "integrations/mason"


def git(repo: str, ref: str) -> dict:
    """Pin to a git repo@ref — a fork/branch (`--repo`/`--ref`) or a recorded Git install."""
    source = repo.removeprefix("git+")
    if "://" not in source:  # a bare local path -> a file:// URL uv can clone
        source = pathlib.Path(source).resolve().as_uri()
    return {"git": source, "rev": ref, "subdirectory": _SUBDIRECTORY}


def editable(mason_dir: pathlib.Path) -> dict:
    """Pin to a local editable path — a live link to a checkout for the dev loop.

    Resolves only on this machine, so `mason deploy` rejects it.
    """
    return {"path": str(mason_dir.resolve()), "editable": True}


def read(pyproject: pathlib.Path) -> dict | None:
    """The `[tool.uv.sources] databricks-mason` table in `pyproject`, or None if absent/unreadable."""
    if not pyproject.is_file():
        return None
    try:
        data = tomli.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomli.TOMLDecodeError):
        return None
    source = data.get("tool", {}).get("uv", {}).get("sources", {}).get("databricks-mason")
    return source if isinstance(source, dict) else None


def write(pyproject: pathlib.Path, source: dict) -> None:
    """Set `[tool.uv.sources] databricks-mason` to `source`, preserving the file's formatting."""
    document = tomlkit.parse(pyproject.read_text())
    if "tool" not in document:
        document["tool"] = tomlkit.table()
    tool = document["tool"]
    if "uv" not in tool:
        tool["uv"] = tomlkit.table()
    uv = tool["uv"]
    if "sources" not in uv:
        uv["sources"] = tomlkit.table()
    table = tomlkit.inline_table()
    table.update(source)
    uv["sources"]["databricks-mason"] = table
    pyproject.write_text(tomlkit.dumps(document))
