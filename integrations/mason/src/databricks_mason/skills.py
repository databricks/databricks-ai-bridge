"""Discover and attach exact Unity Catalog agent skills."""

from __future__ import annotations

import pathlib
from typing import Any

import click

from databricks_mason import render
from databricks_mason.agent_project import AgentProject, SkillSpec
from databricks_mason.errors import AgentCliError

_RESOURCE_PREFIX = "skills/"
_METADATA_FIELDS = ("bundle_name", "description", "etag", "comment", "effective_owner")


def _validate_schema(schema: str) -> str:
    normalized = schema.strip()
    parts = normalized.split(".")
    if (
        len(parts) != 2
        or any(not part for part in parts)
        or any(character.isspace() for character in normalized)
    ):
        raise AgentCliError(
            f"Invalid schema {schema!r}.",
            hint="Use a two-part Unity Catalog schema name: catalog.schema.",
        )
    return normalized


def _skill_record(skill: Any) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(skill, dict):
        return None, "entry is not an object"
    raw_name = skill.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        return None, "name is missing or is not a string"
    if not raw_name.startswith(_RESOURCE_PREFIX):
        return None, "name must use skills/catalog.schema.skill"
    name = raw_name.removeprefix(_RESOURCE_PREFIX)
    parts = name.split(".")
    if (
        len(parts) != 3
        or any(not part for part in parts)
        or any(character.isspace() for character in name)
    ):
        return None, "name must use skills/catalog.schema.skill"
    record = {"name": name}
    for field in _METADATA_FIELDS:
        value = skill.get(field)
        if isinstance(value, str) and value:
            record[field] = value
    return record, None


def _warn_malformed_skill(page: int, index: int, reason: str) -> None:
    click.echo(
        f"Warning: skipped malformed UC skill entry at page {page}, index {index}: {reason}.",
        err=True,
    )


def _list_uc_skills(client: Any, schema: str) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    page_token = None
    seen_tokens: set[str] = set()
    page_number = 0
    while True:
        page_number += 1
        response = client.list_uc_skills(schema, page_token=page_token)
        if not isinstance(response, dict):
            raise AgentCliError("The Skills API returned an invalid response.")
        page = response.get("skills", [])
        if not isinstance(page, list):
            raise AgentCliError("The Skills API returned an invalid response.")
        for index, skill in enumerate(page, 1):
            record, reason = _skill_record(skill)
            if reason is not None:
                _warn_malformed_skill(page_number, index, reason)
            if record is not None and record["name"] not in by_name:
                by_name[record["name"]] = record
        if "next_page_token" not in response:
            break
        page_token = response["next_page_token"]
        if not isinstance(page_token, str):
            raise AgentCliError(
                "The Skills API returned an invalid response: pagination token must be a string."
            )
        if not page_token:
            break
        if page_token in seen_tokens:
            raise AgentCliError("The Skills API returned a repeated pagination token.")
        seen_tokens.add(page_token)
    return [by_name[name] for name in sorted(by_name)]


def _manifest_record(spec: SkillSpec) -> dict[str, str]:
    assert spec.source.name is not None
    return {
        "id": spec.id,
        "kind": spec.source.kind,
        "source": spec.source.name,
    }


def _emit_change(
    obj: Any,
    project: AgentProject,
    spec: SkillSpec,
    changed_files: list[pathlib.Path],
) -> None:
    payload = {
        "schema_version": 1,
        "changed": bool(changed_files),
        "changed_files": [str(path) for path in changed_files],
        "skill": _manifest_record(spec),
    }
    if getattr(obj, "output", "text") == "json":
        render.emit_json(payload)
        return
    if changed_files:
        render.success(
            f"Added {spec.id}",
            fields={"Kind": spec.source.kind, "Manifest": str(project.path)},
        )
    else:
        click.echo(f"Skill {spec.id!r} is already configured in {project.path}")


def _add_spec(obj: Any, source: pathlib.Path, spec: SkillSpec) -> None:
    project = AgentProject.load(source)
    changed = project.add_skill(spec)
    changed_files = [project.write()] if changed else []
    _emit_change(obj, project, spec, changed_files)


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Mason agent project containing agent.toml.",
    )(function)


@click.group()
def skills() -> None:
    """Discover and attach Unity Catalog agent skills."""


@skills.command("list")
@click.option(
    "--schema",
    required=True,
    help="Two-part Unity Catalog schema containing skills.",
)
@click.pass_obj
def list_skills(obj: Any, schema: str) -> None:
    """List Unity Catalog skills that can be attached to an agent."""
    schema = _validate_schema(schema)
    records = _list_uc_skills(obj.client(), schema)
    if getattr(obj, "output", "text") == "json":
        render.emit_json({"schema_version": 1, "skills": records})
        return
    render.resource_table(
        "Unity Catalog skills",
        [("Skill", "left"), ("Description", "left")],
        [(record["name"], record.get("description")) for record in records],
        subtitle=f"Available in {schema}",
    )
    if records:
        click.echo("\nAdd a skill:")
        for record in records:
            click.echo(f"mason skills add uc {record['name']}")


@skills.group("add")
def add() -> None:
    """Attach an exact skill binding to agent.toml."""


@add.command("uc")
@click.argument("name")
@click.option("--name", "skill_id", default=None)
@_source_option
@click.pass_obj
def add_uc(
    obj: Any,
    name: str,
    skill_id: str | None,
    source: pathlib.Path,
) -> None:
    """Attach an exact three-part Unity Catalog skill."""
    _add_spec(
        obj,
        source,
        SkillSpec.uc(skill_id or name.rsplit(".", 1)[-1], name=name),
    )
