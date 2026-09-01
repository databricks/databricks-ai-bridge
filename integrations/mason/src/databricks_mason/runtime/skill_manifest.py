"""Read the UC skill bindings in the project's ``agent.toml``."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any, cast

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib

from databricks_mason.runtime.tool_manifest import project_root

_SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SkillRecord:
    """One exact Unity Catalog skill binding."""

    id: str
    kind: str
    name: str


def _required_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"agent.toml must declare {description}.")
    return value


def _three_part_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 3 or any(not part for part in parts) or any(char.isspace() for char in value):
        raise RuntimeError(f"Invalid UC skill name {value!r}.")
    return value


def _skill(value: object) -> SkillRecord:
    if not isinstance(value, dict):
        raise RuntimeError("agent.toml skills must be tables.")
    value = cast(dict[str, Any], value)
    skill_id = _required_string(value.get("id"), "a skill id")
    if not _SKILL_ID.fullmatch(skill_id):
        raise RuntimeError(f"Invalid skill id {skill_id!r}.")
    source = value.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("Each agent.toml skill must declare a source table.")
    source = cast(dict[str, Any], source)
    kind = _required_string(source.get("kind"), "a skill source kind")
    if kind != "uc":
        raise RuntimeError(f"Unsupported agent.toml skill kind: {kind!r}; only 'uc' is supported.")
    if source.get("path") is not None:
        raise RuntimeError("UC skill bindings do not accept source.path.")
    name = _three_part_name(_required_string(source.get("name"), "a UC skill source.name"))
    return SkillRecord(id=skill_id, kind=kind, name=name)


def load_skills(*, expected_framework: str) -> tuple[SkillRecord, ...]:
    """Load a fresh immutable view so direct manifest edits apply on the next request."""
    path: pathlib.Path = project_root() / "agent.toml"
    try:
        with path.open("rb") as input_file:
            document: dict[str, Any] = tomllib.load(input_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported agent.toml schema in {path}; expected schema_version = 1.")
    agent = document.get("agent")
    if not isinstance(agent, dict) or agent.get("framework") != expected_framework:
        actual = agent.get("framework") if isinstance(agent, dict) else None
        raise RuntimeError(
            f"agent.toml framework {actual!r} does not match runtime {expected_framework!r}."
        )
    raw_skills = document.get("skills", [])
    if not isinstance(raw_skills, list):
        raise RuntimeError("agent.toml skills must be an array of tables.")
    skills = tuple(_skill(item) for item in raw_skills)
    ids = [skill.id for skill in skills]
    if len(ids) != len(set(ids)):
        raise RuntimeError("agent.toml skill ids must be unique.")
    return skills
