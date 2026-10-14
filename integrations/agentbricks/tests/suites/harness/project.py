"""One test's Agent Bricks project: its run-scoped name and path, its files, and its runtime pin.

Tests create the project themselves with ``agentbricks init`` and then call ``pin_runtime``, so the
project runs the Agent Bricks runtime under test instead of the release on PyPI.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import tomli
import tomlkit

from harness.logs import log
from harness.run import APP_PREFIX, FRAMEWORKS, RunConfig, project_prefix

_DIRECT_HEADER = (
    'schema_version = 1\n\n[agent]\nframework = "{framework}"\nserver = "agentbricks"\n'
)


class ToolSpec(Protocol):
    """What ``write_manifest`` needs from a tool: its [[tools]] TOML and any user-owned files."""

    toml: str
    files: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class Project:
    path: pathlib.Path
    app_name: str
    # The runtime under test: the wheel built from this checkout, or a bridge commit.
    wheel: pathlib.Path | None = None
    bridge_sha: str | None = None

    @classmethod
    def allocate(cls, run_config: RunConfig) -> Project:
        """A unique project path and App name under the run's prefix; nothing is created yet."""
        name = f"{project_prefix(run_config.run_id)}{uuid.uuid4().hex[:6]}"
        path = run_config.output / "projects" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, f"{APP_PREFIX}{name}", run_config.wheel, run_config.bridge_sha)

    def pin_runtime(self) -> None:
        """Point the project's databricks-agentbricks dependency at the runtime under test."""
        if self.bridge_sha:
            _pin_bridge_sources(self.path, self.bridge_sha)
        else:
            assert self.wheel is not None
            _pin_project_wheel(self.path, self.wheel)

    def is_pinned(self) -> bool:
        pyproject = self.path / "pyproject.toml"
        document = tomli.loads(pyproject.read_text(encoding="utf-8"))
        sources = document.get("tool", {}).get("uv", {}).get("sources", {})
        return "databricks-agentbricks" in sources

    def manifest(self) -> dict[str, Any]:
        return tomli.loads((self.path / "agent.toml").read_text(encoding="utf-8"))

    def declared_store(self, kind: str) -> str | None:
        """The name agent.toml gives the ``memory`` or ``session`` store, if it declares one."""
        if not (self.path / "agent.toml").exists():
            return None
        return self.manifest().get(f"{kind}_store", {}).get("name")

    def write_file(self, relative_path: str, content: str) -> None:
        """Write a user-owned file in the project, as a developer would."""
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"# write {target}: user-owned file")
        target.write_text(content, encoding="utf-8")

    def write_files(self, files: Mapping[str, str]) -> None:
        for relative_path, content in files.items():
            self.write_file(relative_path, content)

    def write_manifest(self, tool_specs: Sequence[ToolSpec]) -> None:
        """Direct authoring: write agent.toml with exactly these tools plus a tracing binding."""
        sections: list[str] = [_DIRECT_HEADER.format(framework=FRAMEWORKS[0])]
        sections.extend(spec.toml for spec in tool_specs if spec.toml)
        # Follows default_experiment_name's shape so direct authoring carries the same tracing
        # binding `agentbricks init` gives CLI authoring.
        slug = re.sub(r"[^a-z0-9-]+", "-", self.path.name.lower()).strip("-") or "agent"
        sections.append(f'[tracing]\nexperiment_name = "/Shared/agentbricks_traces/{slug}"\n')
        target = self.path / "agent.toml"
        log(f"# write {target}: direct authoring; no agentbricks tools command")
        target.write_text("\n".join(sections), encoding="utf-8")
        for spec in tool_specs:
            self.write_files(spec.files)


def _pin_project_wheel(project: pathlib.Path, wheel: pathlib.Path) -> None:
    vendor_dir = project / "agentbricks_e2e_wheels"
    vendor_dir.mkdir()
    vendored_wheel = vendor_dir / wheel.name
    shutil.copy2(wheel, vendored_wheel)
    pyproject = project / "pyproject.toml"
    document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    tool = document.setdefault("tool", {})
    uv = tool.setdefault("uv", {})
    sources = uv.setdefault("sources", {})
    sources["databricks-agentbricks"] = {"path": vendored_wheel.relative_to(project).as_posix()}
    log(
        f"# write {pyproject}: "
        f"pin databricks-agentbricks runtime to {vendored_wheel.relative_to(project)}"
    )
    pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")


def _pin_bridge_sources(project: pathlib.Path, bridge_sha: str) -> None:
    pyproject = project / "pyproject.toml"
    document = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    dependencies = document["project"]["dependencies"]
    if not any(str(dependency).startswith("databricks-langchain") for dependency in dependencies):
        dependencies.append("databricks-langchain>=0.17.0")
    if "tool" not in document:
        document["tool"] = tomlkit.table()
    if "uv" not in document["tool"]:
        document["tool"]["uv"] = tomlkit.table()
    if "sources" not in document["tool"]["uv"]:
        document["tool"]["uv"]["sources"] = tomlkit.table()
    for package, subdirectory in (
        ("databricks-agentbricks", "integrations/agentbricks"),
        ("databricks-langchain", "integrations/langchain"),
    ):
        source = tomlkit.inline_table()
        source.update(
            {
                "git": "https://github.com/databricks/databricks-ai-bridge.git",
                "rev": bridge_sha,
                "subdirectory": subdirectory,
            }
        )
        document["tool"]["uv"]["sources"][package] = source
    log(f"# write {pyproject}: pin Agent Bricks and LangChain to {bridge_sha}")
    pyproject.write_text(tomlkit.dumps(document), encoding="utf-8")
