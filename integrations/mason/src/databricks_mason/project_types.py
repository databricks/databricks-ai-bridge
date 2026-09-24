"""Framework and server selections shared by Mason's CLI and project manifests."""

from __future__ import annotations

from enum import Enum

from databricks_mason.errors import AgentCliError


class AgentFramework(str, Enum):
    """Frameworks supported by Mason's project templates and adapters."""

    LANGGRAPH = "langgraph"
    OPENAI = "openai"

    def __str__(self) -> str:
        return self.value


class AgentServer(str, Enum):
    """HTTP server ownership selected when a Mason project is created."""

    MASON = "mason"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


def parse_framework(value: object) -> AgentFramework:
    """Validate a framework selection without exposing Enum errors to CLI users."""
    if isinstance(value, str):
        try:
            return AgentFramework(value)
        except ValueError:
            pass
    rendered = repr(str(value)) if isinstance(value, str) else "missing"
    raise AgentCliError(
        f"Unsupported Agent Bricks framework {rendered}.",
        hint=f"Supported frameworks: {', '.join(framework.value for framework in AgentFramework)}.",
    )


def parse_server(value: object) -> AgentServer:
    """Validate a server selection without exposing Enum errors to CLI users."""
    if isinstance(value, str):
        try:
            return AgentServer(value)
        except ValueError:
            pass
    rendered = repr(str(value)) if isinstance(value, str) else "missing"
    raise AgentCliError(
        f"Unsupported Agent Bricks server {rendered}.",
        hint=f"Supported servers: {', '.join(server.value for server in AgentServer)}.",
    )
