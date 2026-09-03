"""Discover the Mason command tree and render help for any nested command."""

from __future__ import annotations

from collections.abc import Iterator

import click

CommandPath = tuple[str, ...]

_EXAMPLES: dict[CommandPath, tuple[str, ...]] = {
    (): (
        "mason login --profile <profile>",
        "mason init my-agent",
        "cd my-agent",
        "mason dev",
        "mason deploy my-agent",
    ),
    ("login",): ("mason login --profile <profile>",),
    ("logout",): ("mason logout",),
    ("init",): ("mason init my-agent",),
    ("dev",): ("mason dev",),
    ("memory",): ("mason memory stores list", "mason memory entries search --help"),
    ("memory", "stores"): ("mason memory stores list",),
    ("memory", "stores", "create"): ("mason memory stores create --display-name agent-memory",),
    ("memory", "stores", "list"): ("mason memory stores list",),
    ("memory", "stores", "get"): ("mason memory stores get <store>",),
    ("memory", "stores", "update"): (
        'mason memory stores update <store> --description "Agent memory"',
    ),
    ("memory", "stores", "delete"): ("mason memory stores delete <store>",),
    ("memory", "entries"): ("mason memory entries list --store <store> --actor-id alice",),
    ("memory", "entries", "create"): (
        "mason memory entries create --store <store> --actor-id alice "
        '--path /preferences/style.md --content "Terse, code first."',
    ),
    ("memory", "entries", "get"): ("mason memory entries get --store <store> <entry>",),
    ("memory", "entries", "list"): ("mason memory entries list --store <store> --actor-id alice",),
    ("memory", "entries", "search"): (
        'mason memory entries search --store <store> --actor-id alice --query "style"',
    ),
    ("memory", "entries", "update"): (
        'mason memory entries update --store <store> <entry> --content "Concise"',
    ),
    ("memory", "entries", "delete"): ("mason memory entries delete --store <store> <entry>",),
    ("mcp",): ("mason mcp list", "mason mcp list --schema main.tools"),
    ("mcp", "list"): ("mason mcp list", "mason mcp list --schema main.tools"),
    ("sessions",): ("mason sessions stores list", "mason sessions list --help"),
    ("sessions", "stores"): ("mason sessions stores list",),
    ("sessions", "stores", "create"): ("mason sessions stores create --name agent-sessions",),
    ("sessions", "stores", "list"): ("mason sessions stores list",),
    ("sessions", "stores", "get"): ("mason sessions stores get agent-sessions",),
    ("sessions", "stores", "update"): (
        'mason sessions stores update agent-sessions --description "Agent sessions"',
    ),
    ("sessions", "stores", "delete"): ("mason sessions stores delete agent-sessions",),
    ("sessions", "items"): (
        "mason sessions items list --store agent-sessions --session-id <session-id>",
    ),
    ("sessions", "items", "list"): (
        "mason sessions items list --store agent-sessions --session-id <session-id>",
    ),
    ("sessions", "items", "append"): (
        "mason sessions items append --store agent-sessions --session-id <session-id> "
        '--data \'{"role":"user","content":"Hello"}\'',
    ),
    ("sessions", "items", "pop"): (
        "mason sessions items pop --store agent-sessions --session-id <session-id>",
    ),
    ("sessions", "items", "clear"): (
        "mason sessions items clear --store agent-sessions --session-id <session-id>",
    ),
    ("sessions", "create"): ("mason sessions create --store agent-sessions --actor-id alice",),
    ("sessions", "list"): ("mason sessions list --store agent-sessions",),
    ("sessions", "get"): ("mason sessions get <session-id> --store agent-sessions",),
    ("sessions", "update"): (
        "mason sessions update <session-id> --store agent-sessions "
        '--metadata \'{"status":"reviewed"}\'',
    ),
    ("sessions", "delete"): ("mason sessions delete <session-id> --store agent-sessions",),
    ("sessions", "fork"): (
        "mason sessions fork --store agent-sessions --source-session-id <session-id> "
        "--actor-id alice",
    ),
    ("tracing",): ("mason tracing setup --catalog main --schema agent_traces",),
    ("tracing", "setup"): ("mason tracing setup --catalog main --schema agent_traces",),
    ("tracing", "list"): ("mason tracing list --experiment /Users/me/mason-traces/my-agent",),
    ("tracing", "get"): ("mason tracing get <trace-id>",),
    ("tracing", "instrument"): ("mason tracing instrument --destination main.agent_traces",),
    ("deploy",): ("mason deploy my-agent",),
    ("deployments",): ("mason deployments list",),
    ("deployments", "list"): ("mason deployments list",),
    ("deployments", "get"): ("mason deployments get mason-my-agent",),
    ("deployments", "logs"): ("mason deployments logs mason-my-agent",),
    ("deployments", "start"): ("mason deployments start mason-my-agent",),
    ("deployments", "stop"): ("mason deployments stop mason-my-agent",),
    ("deployments", "delete"): ("mason deployments delete mason-my-agent",),
    ("tools",): (
        "mason tools add --help",
        "mason tools add sandbox --scope table:samples.nyctaxi.trips",
        "mason tools add mcp system.ai.web_search",
        "mason tools remove web_search",
        "mason tools list",
    ),
    ("tools", "add"): (
        "mason tools add sandbox --scope table:samples.nyctaxi.trips",
        "mason tools add mcp system.ai.web_search",
        "mason tools add uc-function catalog.schema.lookup_ticket",
        "mason tools add python lookup-ticket",
    ),
    ("tools", "add", "sandbox"): ("mason tools add sandbox --scope table:samples.nyctaxi.trips",),
    ("tools", "add", "mcp"): ("mason tools add mcp system.ai.web_search",),
    ("tools", "add", "uc-function"): ("mason tools add uc-function catalog.schema.lookup_ticket",),
    ("tools", "add", "python"): ("mason tools add python lookup-ticket",),
    ("tools", "remove"): (
        "mason tools remove mcp system.ai.web_search",
        "mason tools remove web_search",
    ),
    ("tools", "list"): ("mason tools list",),
}


def _walk(
    command: click.Command, prefix: CommandPath = ()
) -> Iterator[tuple[CommandPath, click.Command]]:
    if not isinstance(command, click.Group):
        return
    for name, child in command.commands.items():
        path = (*prefix, name)
        yield path, child
        yield from _walk(child, path)


def _example_epilog(examples: tuple[str, ...]) -> str:
    return "\n".join(("\b", "Examples:", *(f"  {example}" for example in examples)))


def configure_help(root: click.Group) -> None:
    """Attach curated examples to the root and every existing command."""
    root.epilog = _example_epilog(_EXAMPLES[()])
    for path, command in _walk(root):
        examples = _EXAMPLES.get(path)
        if examples:
            command.epilog = _example_epilog(examples)
