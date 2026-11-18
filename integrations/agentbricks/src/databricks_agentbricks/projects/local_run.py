"""Resolve a local start command from agent.toml or an existing Apps manifest."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import tomli

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.app_manifest import AppManifest, is_placeholder_command


@dataclass(frozen=True)
class LocalRun:
    command: tuple[str, ...]
    environment: dict[str, str | None]


def _command(value: object, source: Path) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = shlex.split(value)
        except ValueError as exc:
            raise AgentCliError(f"Invalid start command in {source}.") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(part, str) or not part for part in value)
        or is_placeholder_command(value)
    ):
        raise AgentCliError(
            f"A non-empty start command is required in {source}.",
            hint='Set [dev].command = ["uv", "run", "start-server"] in agent.toml, '
            "or provide command: in app.yaml for an existing project.",
            error_category="ERROR_CATEGORY_CONFIGURATION",
        )
    return tuple(str(part) for part in value)


def load_local_run(source: Path) -> LocalRun:
    path = source / "agent.toml"
    if path.is_file():
        try:
            manifest = tomli.loads(path.read_text())
        except (OSError, tomli.TOMLDecodeError) as exc:
            raise AgentCliError(f"Could not read local configuration in {path}.") from exc
        if "dev" in manifest:
            dev = manifest["dev"]
            if not isinstance(dev, dict):
                raise AgentCliError("agent.toml [dev] must be a table.")
            command = dev.get("command")
            if not isinstance(command, list):
                raise AgentCliError(
                    "agent.toml [dev].command must be a non-empty array of strings."
                )
            env = dev.get("env", {})
            if not isinstance(env, dict) or any(
                not isinstance(value, str) for value in env.values()
            ):
                raise AgentCliError("agent.toml [dev.env] must contain string values.")
            return LocalRun(_command(command, path), dict(env))
    path = source / "app.yaml"
    if not path.is_file():
        raise AgentCliError(
            "No local start command found in agent.toml or app.yaml.",
            hint="Run `agentbricks init`, or add [dev].command to agent.toml.",
            error_category="ERROR_CATEGORY_CONFIGURATION",
        )
    manifest = AppManifest.parse(path.read_text(), source=path)
    env = {}
    for entry in manifest.raw_env():
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise AgentCliError("app.yaml env entries must declare a name and value or valueFrom.")
        name = entry["name"]
        value = entry.get("value")
        if "value" in entry and isinstance(value, (str, int, float, bool)):
            env[name] = str(value)
        elif "valueFrom" in entry:
            env[name] = None
        else:
            raise AgentCliError(f"app.yaml env entry {name!r} needs a local value.")
    return LocalRun(_command(manifest.command, path), env)
