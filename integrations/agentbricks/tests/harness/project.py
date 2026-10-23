"""One test's Agent Bricks project: its run-scoped name and path, its files, and its runtime pin.

Tests create the project themselves with ``agentbricks init`` and then call ``pin_runtime``, so the
project runs the Agent Bricks runtime under test instead of the release on PyPI.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import uuid
from collections.abc import Mapping

import tomli
import tomlkit

from harness.logs import log
from harness.run import APP_PREFIX, E2EError, RunConfig, project_prefix


@dataclasses.dataclass(frozen=True)
class Project:
    path: pathlib.Path
    app_name: str
    # The runtime under test: the wheel built from this checkout, or the one --wheel gives.
    wheel: pathlib.Path

    @classmethod
    def allocate(cls, run_config: RunConfig) -> Project:
        """A unique project path and App name under the run's prefix; nothing is created yet."""
        name = f"{project_prefix(run_config.run_id)}{uuid.uuid4().hex[:6]}"
        path = run_config.output / "projects" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if run_config.wheel is None:
            # Only the run_config fixture builds it; the bare run config is for cleanup.
            raise E2EError("The run has no runtime wheel to pin projects to.")
        return cls(path, f"{APP_PREFIX}{name}", run_config.wheel)

    def pin_runtime(self) -> None:
        """Point the project's databricks-agentbricks dependency at the runtime under test."""
        _pin_project_wheel(self.path, self.wheel)

    def is_pinned(self) -> bool:
        pyproject = self.path / "pyproject.toml"
        document = tomli.loads(pyproject.read_text(encoding="utf-8"))
        sources = document.get("tool", {}).get("uv", {}).get("sources", {})
        return "databricks-agentbricks" in sources

    def write_file(self, relative_path: str, content: str) -> None:
        """Write a user-owned file in the project, as a developer would."""
        target = self.path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"# write {target}: user-owned file")
        target.write_text(content, encoding="utf-8")

    def write_files(self, files: Mapping[str, str]) -> None:
        for relative_path, content in files.items():
            self.write_file(relative_path, content)


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
