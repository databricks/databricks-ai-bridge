"""Persistent Mason project metadata and legacy framework detection."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import tomli

from databricks_mason.errors import AgentCliError
from databricks_mason.project_types import AgentFramework, parse_framework

_CONFIG_PATH = pathlib.Path(".mason/project.toml")
_SCHEMA_VERSION = 1
_CUSTOM_SERVER_TEMPLATES = frozenset({"custom-agent-langgraph", "custom-agent-openai"})


@dataclass(frozen=True)
class ProjectMetadata:
    """The template identity persisted by ``mason init``."""

    framework: AgentFramework
    template: str | None
    request_auth_contract_version: int | None = None


def write_project_metadata(
    project: pathlib.Path,
    *,
    framework: str,
    template: str,
    request_auth_contract_version: int | None = None,
) -> pathlib.Path:
    """Write the metadata consumed by template-aware Mason commands."""
    selected_framework = parse_framework(framework)
    target = project / _CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"schema_version = {_SCHEMA_VERSION}\n"
        f"framework = {json.dumps(selected_framework.value)}\n"
        f"template = {json.dumps(template)}\n"
        + (
            f"request_auth_contract_version = {request_auth_contract_version}\n"
            if request_auth_contract_version is not None
            else ""
        ),
        encoding="utf-8",
    )
    return target


def _read_toml(path: pathlib.Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as input_file:
            value = tomli.load(input_file)
    except (OSError, tomli.TOMLDecodeError) as exc:
        raise AgentCliError(f"Could not read {description} at {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise AgentCliError(f"{description.capitalize()} at {path} must be a TOML table.")
    return value


def _load_persisted_metadata(project: pathlib.Path) -> ProjectMetadata | None:
    path = project / _CONFIG_PATH
    if not path.is_file():
        return None
    data = _read_toml(path, "Mason project config")
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise AgentCliError(
            f"Unsupported Mason project config schema in {path}.",
            hint=f"Expected schema_version = {_SCHEMA_VERSION}.",
        )
    framework = parse_framework(data.get("framework"))
    template = data.get("template")
    if not isinstance(template, str) or not template:
        raise AgentCliError(f"Mason project config at {path} must declare a template.")
    contract = data.get("request_auth_contract_version")
    if contract is not None and (type(contract) is not int or contract != 1):
        raise AgentCliError(f"Unsupported request_auth_contract_version in {path}.")
    return ProjectMetadata(
        framework=framework, template=template, request_auth_contract_version=contract
    )


def _dependency_name(requirement: str) -> str:
    """Return a normalized distribution name from a PEP 508 requirement."""
    name = re.split(r"[\s\[<>=!~;@]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _infer_legacy_framework(project: pathlib.Path) -> ProjectMetadata:
    pyproject = project / "pyproject.toml"
    if not pyproject.is_file():
        raise AgentCliError(
            f"Could not determine the Mason framework for {project}.",
            hint="Run `mason init` to create project metadata or pass `--framework`.",
        )
    data = _read_toml(pyproject, "pyproject")
    project_table = data.get("project")
    dependencies = project_table.get("dependencies", []) if isinstance(project_table, dict) else []
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        dependencies = []
    packages = {_dependency_name(item) for item in dependencies}
    candidates = {
        framework
        for package, framework in (
            ("databricks-openai", AgentFramework.OPENAI),
            ("databricks-langchain", AgentFramework.LANGGRAPH),
        )
        if package in packages
    }
    if len(candidates) != 1:
        detail = (
            "both framework dependencies are present"
            if candidates
            else "no framework dependency was found"
        )
        raise AgentCliError(
            f"Could not determine the Mason framework for {project}: {detail}.",
            hint="Pass `--framework openai` or `--framework langgraph`.",
        )
    return ProjectMetadata(framework=candidates.pop(), template=None)


def load_project_metadata(
    project: pathlib.Path,
    *,
    framework_override: str | None = None,
) -> ProjectMetadata:
    """Load init metadata, with dependency inference for projects created before it existed."""
    persisted = _load_persisted_metadata(project)
    if framework_override is not None:
        override = parse_framework(framework_override)
        if persisted is not None and persisted.framework != override:
            raise AgentCliError(
                f"Framework override '{override.value}' conflicts with {persisted.framework.value!r} in "
                f"{project / _CONFIG_PATH}."
            )
        return persisted or ProjectMetadata(framework=override, template=None)
    return persisted or _infer_legacy_framework(project)


def is_custom_server_template(template: str | None) -> bool:
    """Whether ``template`` is one of Mason's custom HTTP server templates."""
    return template in _CUSTOM_SERVER_TEMPLATES


def uses_custom_server(project: pathlib.Path) -> bool:
    """Whether persisted project metadata selects a Mason custom server template."""
    metadata_path = project / _CONFIG_PATH
    if not metadata_path.is_file():
        return False
    return is_custom_server_template(load_project_metadata(project).template)


def require_managed_tool_support(project: pathlib.Path) -> None:
    """Reject managed tool bindings for known templates that do not consume them."""
    if not uses_custom_server(project):
        return
    raise AgentCliError(
        "Managed tool bindings in agent.toml require a Mason server template.",
        hint="Wire tools directly in agent/agent.py, or create a project with "
        "`mason init --server mason`.",
    )
