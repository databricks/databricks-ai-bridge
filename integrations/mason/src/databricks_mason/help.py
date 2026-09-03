"""Discover the Mason command tree and render help for any nested command."""

from __future__ import annotations

from collections.abc import Iterator

import click

CommandPath = tuple[str, ...]

# Each example is either a bare command, or a (command, comment) pair. The comment is a short gloss
# rendered beside/above the command so a first-time reader can tell what each example does without
# running it. See `_example_epilog` for how comments are laid out.
Example = str | tuple[str, str]

_EXAMPLES: dict[CommandPath, tuple[Example, ...]] = {
    (): (
        ("mason login --profile <profile>", "authenticate and save a default profile"),
        ("mason init my-agent", "scaffold a new agent project"),
        ("cd my-agent", "enter the project directory"),
        ("mason dev", "run the agent locally with a chat UI"),
        ("mason deploy my-agent", "deploy the agent to Databricks Apps"),
    ),
    ("login",): (("mason login --profile <profile>", "save a profile as your default"),),
    ("logout",): (("mason logout", "forget the saved default profile"),),
    ("init",): (("mason init my-agent", "scaffold a new agent project"),),
    ("dev",): (("mason dev", "run the agent locally with a chat UI"),),
    ("memory",): (
        ("mason memory stores create --display-name agent-memory", "create a memory store"),
        ("mason deploy <agent> --memory agent-memory", "wire it into a deployment"),
    ),
    ("memory", "stores"): (("mason memory stores list", "list managed memory stores"),),
    ("memory", "stores", "create"): (
        ("mason memory stores create --display-name agent-memory", "create a memory store"),
    ),
    ("memory", "stores", "list"): (("mason memory stores list", "list managed memory stores"),),
    ("memory", "stores", "get"): (("mason memory stores get <store>", "show one store's details"),),
    ("memory", "stores", "update"): (
        ('mason memory stores update <store> --description "Agent memory"', "edit a store"),
    ),
    ("memory", "stores", "delete"): (("mason memory stores delete <store>", "delete a store"),),
    ("memory", "entries"): (
        ("mason memory entries list --store <store> --actor-id alice", "list an actor's entries"),
    ),
    ("memory", "entries", "create"): (
        (
            "mason memory entries create --store <store> --actor-id alice "
            '--path /preferences/style.md --content "Terse, code first."',
            "add a memory entry for an actor",
        ),
    ),
    ("memory", "entries", "get"): (
        ("mason memory entries get --store <store> <entry>", "show one entry"),
    ),
    ("memory", "entries", "list"): (
        ("mason memory entries list --store <store> --actor-id alice", "list an actor's entries"),
    ),
    ("memory", "entries", "search"): (
        (
            'mason memory entries search --store <store> --actor-id alice --query "style"',
            "search an actor's entries",
        ),
    ),
    ("memory", "entries", "update"): (
        (
            'mason memory entries update --store <store> <entry> --content "Concise"',
            "edit an entry",
        ),
    ),
    ("memory", "entries", "delete"): (
        ("mason memory entries delete --store <store> <entry>", "delete an entry"),
    ),
    ("mcp",): (
        ("mason mcp list", "list workspace MCP services"),
        ("mason mcp list --schema main.tools", "scope the list to one UC schema"),
    ),
    ("mcp", "list"): (
        ("mason mcp list", "list workspace MCP services"),
        ("mason mcp list --schema main.tools", "scope the list to one UC schema"),
    ),
    ("sessions",): (
        ("mason sessions stores list", "list managed session stores"),
        ("mason sessions list --help", "see how to list sessions"),
    ),
    ("sessions", "stores"): (("mason sessions stores list", "list managed session stores"),),
    ("sessions", "stores", "create"): (
        ("mason sessions stores create --name agent-sessions", "create a session store"),
    ),
    ("sessions", "stores", "list"): (
        ("mason sessions stores list", "list managed session stores"),
    ),
    ("sessions", "stores", "get"): (
        ("mason sessions stores get agent-sessions", "show one store's details"),
    ),
    ("sessions", "stores", "update"): (
        (
            'mason sessions stores update agent-sessions --description "Agent sessions"',
            "edit a store",
        ),
    ),
    ("sessions", "stores", "delete"): (
        ("mason sessions stores delete agent-sessions", "delete a store"),
    ),
    ("sessions", "items"): (
        (
            "mason sessions items list --store agent-sessions --session-id <session-id>",
            "list a session's items",
        ),
    ),
    ("sessions", "items", "list"): (
        (
            "mason sessions items list --store agent-sessions --session-id <session-id>",
            "list a session's items",
        ),
    ),
    ("sessions", "items", "append"): (
        (
            "mason sessions items append --store agent-sessions --session-id <session-id> "
            '--data \'{"role":"user","content":"Hello"}\'',
            "append an item to a session",
        ),
    ),
    ("sessions", "items", "pop"): (
        (
            "mason sessions items pop --store agent-sessions --session-id <session-id>",
            "remove the last item",
        ),
    ),
    ("sessions", "items", "clear"): (
        (
            "mason sessions items clear --store agent-sessions --session-id <session-id>",
            "remove all items",
        ),
    ),
    ("sessions", "create"): (
        ("mason sessions create --store agent-sessions --actor-id alice", "start a new session"),
    ),
    ("sessions", "list"): (
        ("mason sessions list --store agent-sessions", "list sessions in a store"),
    ),
    ("sessions", "get"): (
        ("mason sessions get <session-id> --store agent-sessions", "show one session"),
    ),
    ("sessions", "update"): (
        (
            "mason sessions update <session-id> --store agent-sessions "
            '--metadata \'{"status":"reviewed"}\'',
            "edit a session's metadata",
        ),
    ),
    ("sessions", "delete"): (
        ("mason sessions delete <session-id> --store agent-sessions", "delete a session"),
    ),
    ("sessions", "fork"): (
        (
            "mason sessions fork --store agent-sessions --source-session-id <session-id> "
            "--actor-id alice",
            "copy a session into a new one",
        ),
    ),
    ("tracing",): (
        ("mason tracing setup --catalog main --schema agent_traces", "link a UC trace destination"),
    ),
    ("tracing", "setup"): (
        ("mason tracing setup --catalog main --schema agent_traces", "link a UC trace destination"),
    ),
    ("tracing", "list"): (
        ("mason tracing list --experiment /Users/me/mason-traces/my-agent", "list recent traces"),
    ),
    ("tracing", "get"): (("mason tracing get <trace-id>", "show one trace"),),
    ("tracing", "instrument"): (
        ("mason tracing instrument --destination main.agent_traces", "print instrumentation code"),
    ),
    ("deploy",): (("mason deploy my-agent", "deploy the agent to Databricks Apps"),),
    ("deployments",): (("mason deployments list", "list agent deployments"),),
    ("deployments", "list"): (("mason deployments list", "list agent deployments"),),
    ("deployments", "get"): (("mason deployments get mason-my-agent", "show one deployment"),),
    ("deployments", "logs"): (
        ("mason deployments logs mason-my-agent", "stream a deployment's logs"),
    ),
    ("deployments", "start"): (("mason deployments start mason-my-agent", "start a deployment"),),
    ("deployments", "stop"): (("mason deployments stop mason-my-agent", "stop a deployment"),),
    ("deployments", "delete"): (
        ("mason deployments delete mason-my-agent", "delete a deployment"),
    ),
    ("tools",): (
        ("mason tools add --help", "see all tool types you can add"),
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
        ("mason tools add mcp system.ai.web_search", "add a managed MCP tool"),
        ("mason tools remove mcp system.ai.web_search", "remove a tool binding"),
        ("mason tools list", "list configured tools"),
    ),
    ("tools", "add"): (
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
        ("mason tools add mcp system.ai.web_search", "add a managed MCP tool"),
        ("mason tools add uc-function catalog.schema.lookup_ticket", "add a UC function tool"),
        ("mason tools add python lookup-ticket", "scaffold a local Python tool"),
    ),
    ("tools", "add", "sandbox"): (
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
    ),
    ("tools", "add", "mcp"): (
        ("mason tools add mcp system.ai.web_search", "add a managed MCP tool"),
    ),
    ("tools", "add", "uc-function"): (
        ("mason tools add uc-function catalog.schema.lookup_ticket", "add a UC function tool"),
    ),
    ("tools", "add", "python"): (
        ("mason tools add python lookup-ticket", "scaffold a local Python tool"),
    ),
    ("tools", "remove"): (
        ("mason tools remove mcp system.ai.web_search", "remove an MCP tool by service"),
        ("mason tools remove web_search", "remove a tool by id"),
    ),
    ("tools", "list"): (("mason tools list", "list configured tools"),),
}

# Longest command we align an inline `# comment` after. Past this, a group's comments would be
# pushed so far right they wrap or scroll off, so we stack the comment on the line above instead.
_INLINE_COMMENT_MAX = 46


def _walk(
    command: click.Command, prefix: CommandPath = ()
) -> Iterator[tuple[CommandPath, click.Command]]:
    if not isinstance(command, click.Group):
        return
    for name, child in command.commands.items():
        path = (*prefix, name)
        yield path, child
        yield from _walk(child, path)


def _split(example: Example) -> tuple[str, str | None]:
    """Normalize an example into (command, comment-or-None)."""
    if isinstance(example, tuple):
        return example[0], example[1]
    return example, None


def _example_epilog(examples: tuple[Example, ...]) -> str:
    """Render the Examples block, keeping commands left-aligned and comments legible.

    Commands sit flush at a two-space indent so the block scans as a clean column. The whole group
    uses one comment layout for consistency: if every command is short enough, comments go inline
    (`cmd  # what it does`) aligned across the group; if any command is long, all comments stack on
    the line above their command so nothing wraps.
    """
    pairs = [_split(example) for example in examples]
    stack = any(comment and len(cmd) > _INLINE_COMMENT_MAX for cmd, comment in pairs)
    inline_width = max((len(cmd) for cmd, comment in pairs if comment), default=0)
    lines = ["\b", "Examples:"]
    for cmd, comment in pairs:
        if not comment:
            lines.append(f"  {cmd}")
        elif stack:
            lines.append(f"  # {comment}")
            lines.append(f"  {cmd}")
        else:
            lines.append(f"  {cmd.ljust(inline_width)}  # {comment}")
    return "\n".join(lines)


def configure_help(root: click.Group) -> None:
    """Attach curated examples to the root and every existing command."""
    root.epilog = _example_epilog(_EXAMPLES[()])
    for path, command in _walk(root):
        examples = _EXAMPLES.get(path)
        if examples:
            command.epilog = _example_epilog(examples)
