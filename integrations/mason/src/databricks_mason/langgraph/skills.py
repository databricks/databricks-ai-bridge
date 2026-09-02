"""Inject declared UC Skills as metadata context and two lazy LangGraph tools."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Any

from databricks_langchain import DatabricksMCPServer
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.sessions import create_session

from databricks_mason.runtime.skill_manifest import SkillRecord, load_skills
from databricks_mason.runtime.workspace import workspace_client, workspace_headers

_MAX_CONTENT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SkillDescriptor:
    """Prompt-safe metadata for one declared UC skill."""

    id: str
    source: str
    name: str
    description: str


def _relative_path(value: str) -> str:
    windows = pathlib.PureWindowsPath(value)
    parts = re.split(r"[/\\]", value)
    if (
        not value
        or pathlib.PurePosixPath(value).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RuntimeError(f"Skill file path {value!r} must be a contained relative path.")
    return pathlib.PurePosixPath(*parts).as_posix()


def _uc_failure(skill_id: str, operation: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Declared skill {skill_id!r} failed at the UC Skills endpoint during {operation}: {exc}."
    )


def _result_text(result: Any, skill_id: str) -> str:
    if getattr(result, "isError", False):
        raise RuntimeError(f"UC Skills endpoint returned an error for declared skill {skill_id!r}.")
    blocks = getattr(result, "content", None)
    texts = (
        [
            item.text
            for item in blocks
            if getattr(item, "type", None) == "text"
            and isinstance(getattr(item, "text", None), str)
        ]
        if isinstance(blocks, (list, tuple))
        else []
    )
    if not texts:
        raise RuntimeError(
            f"Declared skill {skill_id!r} did not return a valid MCP result with a text block "
            "from the UC Skills endpoint."
        )
    content = "\n".join(texts)
    try:
        size = len(content.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"UC skill content for {skill_id!r} must be UTF-8.") from exc
    if size > _MAX_CONTENT_BYTES:
        raise RuntimeError(f"UC skill content for {skill_id!r} exceeds the 1 MiB limit.")
    return content


class _UCProvider:
    def __init__(self, record: SkillRecord, server: DatabricksMCPServer, tool: Any):
        self.record = record
        self.server = server
        description = getattr(tool, "description", None)
        if not isinstance(description, str) or not description.strip():
            raise RuntimeError(
                f"Declared skill {record.id!r} has no description at the UC Skills endpoint."
            )
        description = description.strip()
        provenance_prefix = f"[{record.name}] "
        if description.startswith(provenance_prefix):
            description = description[len(provenance_prefix) :].strip()
        if not description:
            raise RuntimeError(
                f"Declared skill {record.id!r} has no description at the UC Skills endpoint."
            )
        self.tool_name = str(tool.name)
        self.descriptor = SkillDescriptor(
            id=record.id,
            source=record.name,
            name=record.name.rsplit(".", 1)[-1],
            description=description,
        )

    async def _call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            async with create_session(self.server.to_connection_dict()) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        except Exception as exc:
            raise _uc_failure(self.record.id, "content loading", exc) from exc
        return _result_text(result, self.record.id)

    async def load(self) -> str:
        return await self._call(self.tool_name, {})

    async def read_file(self, path: str) -> str:
        relative = _relative_path(path)
        return await self._call(
            "get_skill_files",
            {"full_name": self.record.name, "paths": [relative]},
        )


def _uc_server(catalog: str, schema: str) -> DatabricksMCPServer:
    client = workspace_client()
    host = client.config.host.rstrip("/")
    return DatabricksMCPServer(
        name=f"uc-skills-{catalog}-{schema}",
        url=f"{host}/ai-gateway/skills/?schema={catalog}.{schema}",
        headers=workspace_headers() or None,
        workspace_client=client,
        timeout=120.0,
    )


async def _uc_providers(records: tuple[SkillRecord, ...]) -> list[_UCProvider]:
    groups: dict[tuple[str, str], list[SkillRecord]] = {}
    for record in records:
        catalog, schema, _ = record.name.split(".")
        groups.setdefault((catalog, schema), []).append(record)

    providers: list[_UCProvider] = []
    for (catalog, schema), declarations in groups.items():
        server = _uc_server(catalog, schema)
        try:
            async with create_session(server.to_connection_dict()) as session:
                await session.initialize()
                response = await session.list_tools()
        except Exception as exc:
            raise _uc_failure(declarations[0].id, "metadata discovery", exc) from exc
        tools: dict[str, list[Any]] = {}
        for tool in getattr(response, "tools", ()):
            name = getattr(tool, "name", None)
            if isinstance(name, str):
                tools.setdefault(name, []).append(tool)
        for record in declarations:
            matches = tools.get(f"skill_{record.name}", [])
            if not matches:
                raise RuntimeError(
                    f"Declared skill {record.id!r} was not found at the UC Skills endpoint "
                    f"for {catalog}.{schema}."
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Declared skill {record.id!r} is ambiguous because duplicate dynamic tools "
                    f"were returned by the UC Skills endpoint for {catalog}.{schema}."
                )
            providers.append(_UCProvider(record, server, matches[0]))
    return providers


def _tools(providers: tuple[_UCProvider, ...]) -> list[BaseTool]:
    registry = {provider.descriptor.id: provider for provider in providers}

    def provider_for(skill_id: str) -> _UCProvider:
        provider = registry.get(skill_id)
        if provider is None:
            raise RuntimeError(f"Unknown or undeclared skill ID {skill_id!r}.")
        return provider

    async def load_skill(skill_id: str) -> str:
        """Load the instructions for one declared skill by its ID."""
        return await provider_for(skill_id).load()

    async def read_skill_file(skill_id: str, path: str) -> str:
        """Read a relative file referenced by one declared skill."""
        return await provider_for(skill_id).read_file(path)

    return [
        StructuredTool.from_function(
            coroutine=load_skill,
            name="load_skill",
            description="Load instructions for a declared skill by its ID.",
        ),
        StructuredTool.from_function(
            coroutine=read_skill_file,
            name="read_skill_file",
            description="Read a relative file referenced by a declared skill.",
        ),
    ]


async def build_skill_context() -> tuple[str, list[BaseTool]]:
    """Return metadata-only prompt context plus lazy tools for exact UC skill bindings."""
    records = load_skills(expected_framework="langgraph")
    if not records:
        return "", []
    providers = tuple(await _uc_providers(records))
    lines = [
        f"- [{descriptor.id}] (uc:{descriptor.source}) {descriptor.description}"
        for descriptor in sorted(
            (provider.descriptor for provider in providers), key=lambda item: item.id
        )
    ]
    context = (
        "Available skills:\n"
        + "\n".join(lines)
        + "\n\nCall load_skill with an ID when a task matches. "
        "Read referenced files only with read_skill_file."
    )
    return context, _tools(providers)
