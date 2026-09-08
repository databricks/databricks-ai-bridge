"""Validated user configuration for a generated Mason agent."""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, cast

import tomlkit
from tomlkit.exceptions import ParseError

from databricks_mason.errors import AgentCliError

_SCHEMA_VERSION = 1
_SUPPORTED_FRAMEWORKS = frozenset({"langgraph", "openai"})
_PROJECT_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentCliError(f"Agent {field} cannot be empty.")
    return value


def _atomic_write(path: pathlib.Path, content: str) -> None:
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = pathlib.Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise AgentCliError(f"Could not write agent configuration at {path}: {exc}.") from exc


@dataclass(frozen=True)
class AgentConfig:
    """Framework-neutral configuration collected by ``mason create``."""

    name: str
    framework: str
    model: str
    instructions: str
    chat_app_enabled: bool

    def __post_init__(self) -> None:
        name = _required_text(self.name, "name")
        if not _PROJECT_NAME.fullmatch(name):
            raise AgentCliError(
                f"Invalid agent name {name!r}.",
                hint="Use letters, numbers, dots, underscores, or hyphens.",
            )
        if not isinstance(self.framework, str) or self.framework not in _SUPPORTED_FRAMEWORKS:
            rendered = repr(self.framework) if isinstance(self.framework, str) else "missing"
            raise AgentCliError(
                f"Unsupported Mason framework {rendered}.",
                hint=f"Supported frameworks: {', '.join(sorted(_SUPPORTED_FRAMEWORKS))}.",
            )
        model = _required_text(self.model, "model")
        if any(character.isspace() for character in model):
            raise AgentCliError("Agent model endpoint cannot contain whitespace.")
        _required_text(self.instructions, "instructions")
        if not isinstance(self.chat_app_enabled, bool):
            raise AgentCliError("Agent chat_app_enabled must be a boolean.")

    def write(self, project: pathlib.Path) -> pathlib.Path:
        """Persist this configuration into an existing ``agent.toml``."""
        project_root = pathlib.Path(project).expanduser().resolve()
        path = project_root / "agent.toml"
        try:
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AgentCliError(f"Could not find agent.toml in {project_root}.") from exc
        except (OSError, UnicodeError, ParseError) as exc:
            raise AgentCliError(f"Could not read agent configuration at {path}: {exc}.") from exc

        if document.get("schema_version") != _SCHEMA_VERSION:
            raise AgentCliError(
                f"Unsupported agent configuration schema in {path}.",
                hint=f"Expected schema_version = {_SCHEMA_VERSION}.",
            )
        agent = document.get("agent")
        if not isinstance(agent, MutableMapping):
            raise AgentCliError(f"Agent configuration at {path} must declare an [agent] table.")
        agent = cast(MutableMapping[str, Any], agent)
        agent["name"] = self.name
        agent["framework"] = self.framework
        agent["model"] = self.model
        agent["instructions"] = self.instructions
        agent["chat_app_enabled"] = self.chat_app_enabled

        _atomic_write(path, tomlkit.dumps(document))
        return path
