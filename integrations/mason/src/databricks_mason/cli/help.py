"""Discover the Mason command tree and render help for any nested command."""

from __future__ import annotations

import re
from collections.abc import Iterator

import click

from databricks_mason._group import apply_group_class, dim

CommandPath = tuple[str, ...]

# Root commands grouped by intent, so the top-level `mason --help` reads as a workflow instead of a
# flat alphabetical dump. Ordered SETUP → DEVELOP → SHIP, matching the
# getting-started path. Any command missing here still lists under "Other commands" (see
# `_group.MasonGroup`).
_COMMAND_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SETUP", ("login", "logout", "init")),
    ("DEVELOP", ("dev", "tools", "memory", "sessions", "tracing")),
    ("SHIP", ("deploy", "deployments")),
)

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
        ("mason memory bind agent-memory", "bind it to the agent (wired in on dev / deploy)"),
        (
            "mason memory entries create --store <store> --actor-id alice "
            '--path /preferences/style.md --content "Terse, code first."',
            "add a memory entry for an actor (--store takes the store id)",
        ),
        (
            'mason memory entries search --store <store> --actor-id alice --query "style"',
            "search an actor's entries",
        ),
    ),
    ("memory", "bind"): (
        ("mason memory bind agent-memory --source .", "declare a memory store in agent.toml"),
    ),
    ("memory", "unbind"): (
        ("mason memory unbind --source .", "remove the memory store binding from agent.toml"),
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
        ("mason sessions stores create --name agent-sessions", "create a session store"),
        ("mason sessions bind agent-sessions", "bind it to the agent (wired in on dev / deploy)"),
        (
            "mason sessions create --store agent-sessions --actor-id alice",
            "start a session for an actor",
        ),
        (
            "mason sessions items append --store agent-sessions --session-id <session-id> "
            '--data \'{"role":"user","content":"Hello"}\'',
            "append an item to the session",
        ),
        (
            "mason sessions items list --store agent-sessions --session-id <session-id>",
            "list the session's items",
        ),
    ),
    ("sessions", "bind"): (
        ("mason sessions bind agent-sessions --source .", "declare a session store in agent.toml"),
    ),
    ("sessions", "unbind"): (
        ("mason sessions unbind --source .", "remove the session store binding from agent.toml"),
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
        (
            "mason tracing configure",
            "use the default per-project experiment (tracing is on by default)",
        ),
        ("mason tracing configure --experiment 12345", "trace to a specific experiment id"),
        ("mason tracing disable", "turn tracing off"),
    ),
    ("tracing", "configure"): (
        ("mason tracing configure --experiment 12345", "trace to a specific experiment id"),
    ),
    ("tracing", "disable"): (("mason tracing disable", "turn tracing off"),),
    ("tracing", "list"): (("mason tracing list --experiment 12345", "list recent traces"),),
    ("tracing", "get"): (("mason tracing get <trace-id>", "show one trace"),),
    ("deploy",): (
        ("mason deploy my-agent", "deploy the agent"),
        ("mason deploy my-agent --instances 2", "deploy with two instances"),
    ),
    ("deployments",): (("mason deployments list", "list agent deployments"),),
    ("deployments", "list"): (("mason deployments list", "list agent deployments"),),
    ("deployments", "get"): (
        ("mason deployments get agent-mason-my-agent", "show one deployment"),
    ),
    ("deployments", "logs"): (
        ("mason deployments logs agent-mason-my-agent", "stream a deployment's logs"),
    ),
    ("deployments", "start"): (
        ("mason deployments start agent-mason-my-agent", "start a deployment"),
    ),
    ("deployments", "stop"): (
        ("mason deployments stop agent-mason-my-agent", "stop a deployment"),
    ),
    ("deployments", "delete"): (
        ("mason deployments delete agent-mason-my-agent", "delete a deployment"),
    ),
    ("endpoint",): (
        (
            "mason endpoint invoke agent-mason-my-agent --path /api/invocations "
            "--json "
            '\'{"id":"00000000-0000-4000-8000-000000000001",'
            '"input":[{"role":"user","content":"Hello"}]}\'',
            "invoke a deployed HTTP agent",
        ),
    ),
    ("endpoint", "invoke"): (
        (
            "mason endpoint invoke agent-mason-my-agent --path /api/invocations "
            "--json "
            '\'{"id":"00000000-0000-4000-8000-000000000001",'
            '"input":[{"role":"user","content":"Hello"}]}\'',
            "invoke a deployed HTTP agent",
        ),
        (
            "mason endpoint invoke --url http://localhost:8000 --path /custom/run "
            '--json \'{"input":"hello"}\'',
            "invoke a local or arbitrary HTTP server",
        ),
    ),
    ("tools",): (
        ("mason tools add --help", "see all tool types you can add"),
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
        ("mason tools add mcp system.ai.python_exec", "add a managed MCP tool"),
        ("mason tools remove mcp system.ai.python_exec", "remove a tool binding"),
        ("mason tools list", "browse available integrations to add"),
    ),
    ("tools", "add"): (
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
        ("mason tools add mcp system.ai.python_exec", "add a managed MCP tool"),
        ("mason tools add uc-function catalog.schema.lookup_ticket", "add a UC function tool"),
        ("mason tools add genie-one", "add workspace-wide Genie One tools"),
        ("mason tools add genie-agent SPACE_ID", "add tools for one Genie Space"),
    ),
    ("tools", "add", "sandbox"): (
        ("mason tools add sandbox --scope table:samples.nyctaxi.trips", "add a data sandbox tool"),
    ),
    ("tools", "add", "mcp"): (
        ("mason tools add mcp system.ai.python_exec", "add a managed MCP tool"),
    ),
    ("tools", "add", "uc-function"): (
        ("mason tools add uc-function catalog.schema.lookup_ticket", "add a UC function tool"),
    ),
    ("tools", "remove"): (
        ("mason tools remove mcp system.ai.python_exec", "remove an MCP tool by service"),
        ("mason tools remove python_exec", "remove a tool by id"),
    ),
    ("tools", "list"): (
        ("mason tools list", "browse built-in recipes and system.ai MCP Services"),
        ("mason tools list --kind mcp", "discover MCP Services in system.ai"),
        (
            "mason tools list --kind mcp --schema main.tools",
            "replace the default MCP schema",
        ),
        ("mason tools list --kind sandbox", "show the local recipe without authentication"),
        ("mason tools list --kind genie-one", "show the Genie One add recipe"),
        ("mason tools list --kind genie-agent", "show the Genie Agent add recipe"),
    ),
}

# Longest command we align an inline `# comment` after. Past this, a group's comments would be
# pushed so far right they wrap or scroll off, so we stack the comment on the line above instead.
_INLINE_COMMENT_MAX = 46

# Where to send a reader who wants more than the help text — best-practice CLI help links out to
# docs and a support/issues path. Shown only on the root `mason --help`, so subcommand help stays
# uncluttered.
_DOCS_URL = "https://github.com/databricks/databricks-ai-bridge/tree/main/integrations/mason"
_ISSUES_URL = "https://github.com/databricks/databricks-ai-bridge/issues"

# Short, one-line descriptors for the root command list. Click renders that list as a scannable
# index and truncates a long first docstring line with `…`; a curated `short_help` keeps each row
# crisp while the command's own `--help` page still shows its full docstring. Keep these under ~45
# chars so they never truncate.
_SHORT_HELP: dict[CommandPath, str] = {
    ("login",): "Authenticate and save a default profile",
    ("logout",): "Forget the saved default profile",
    ("init",): "Scaffold a new agent project",
    ("dev",): "Run the agent locally with a chat UI",
    ("deploy",): "Deploy an agent to Databricks Apps",
    ("deployments",): "Manage deployed agents",
    ("memory",): "Manage an agent's long-term memory",
    ("sessions",): "Manage an agent's conversation sessions",
    ("mcp",): "Deprecated MCP discovery alias",
    ("tools",): "Discover integrations and manage tool bindings",
    ("tracing",): "Set up and inspect agent tracing",
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
    # Heading grayed + uppercased to match the formatter-rendered sections (OPTIONS/SETUP/…); the
    # commands stay at the terminal's default foreground and the `# comments` are grayed like every
    # other description on the page. `ljust` is computed on the plain command so alignment survives.
    lines = ["\b", dim("EXAMPLES")]
    for cmd, comment in pairs:
        if not comment:
            lines.append(f"  {cmd}")
        elif stack:
            lines.append(f"  {dim(f'# {comment}')}")
            lines.append(f"  {cmd}")
        else:
            lines.append(f"  {cmd.ljust(inline_width)}  {dim(f'# {comment}')}")
    return "\n".join(lines)


def _getting_started_epilog() -> str:
    """The root's numbered "Getting started" path: login → init → cd → dev → deploy.

    A numbered, ordered path — rather than an unlabeled grab-bag of examples — removes the
    "blank-page problem" for a first-time reader: it says *start here, in this order*. Reuses the
    root happy-path examples so the path and their glosses stay in one place.
    """
    pairs = [_split(example) for example in _EXAMPLES[()]]
    width = max(len(cmd) for cmd, _ in pairs)
    lines = ["\b", dim("GETTING STARTED")]
    for i, (cmd, comment) in enumerate(pairs, start=1):
        row = f"  {i}  {cmd.ljust(width)}"
        if comment:
            row += f"  {dim(f'# {comment}')}"
        lines.append(row)
    return "\n".join(lines)


def _root_epilog() -> str:
    """The root help footer: the numbered getting-started path, an auth note, then Docs/Issues links.

    Each block is its own `\\b` paragraph so Click renders it verbatim (commands and URLs intact)
    instead of rewrapping it.
    """
    # The auth note (setup instructions) and the Docs/Issues links stay at full intensity — this is
    # important standalone content, not the secondary command *descriptions* that the graying is for.
    auth = "\n".join(
        [
            "\b",
            "Not authenticated yet? Create a profile with the Databricks CLI first:",
            "  databricks auth login --profile <profile>",
            "Then `mason login --profile <profile>` saves it as your default.",
        ]
    )
    links = "\n".join(["\b", f"Docs:   {_DOCS_URL}", f"Issues: {_ISSUES_URL}"])
    return f"{_getting_started_epilog()}\n\n{auth}\n\n{links}"


# A capabilities row: indent, label (single word), a 2+ space gap, then the description.
_CAPABILITY_ROW = re.compile(r"^(\s*)(\S+)(\s{2,})(.*)$")


def _dim_capabilities_block(help_text: str) -> str:
    """Gray only the *descriptions* in the root docstring's `\\b` capabilities block.

    The block (Models/Tools/…) is a reference list that lives in the docstring, which the formatter
    leaves at full contrast. Match the rest of the help page: the label column (left) stays at full
    weight and the description recedes as secondary. A row is ``indent label  description``; a
    wrapped continuation line has no label, so it dims whole. Leading indent is preserved so
    alignment and Click's dedent are unaffected. Scoped to the single `\\b` verbatim block (marked by
    a lone ``\\x08``), up to the next blank line.
    """
    out: list[str] = []
    dimming = False
    for line in help_text.split("\n"):
        stripped = line.strip()
        if not dimming and stripped == "\x08":
            dimming = True
            out.append(line)
        elif dimming and stripped == "":
            dimming = False
            out.append(line)
        elif dimming:
            row = _CAPABILITY_ROW.match(line)
            if row:
                indent, label, gap, description = row.groups()
                # Label full-weight; only the description recedes.
                out.append(f"{indent}{label}{gap}{dim(description)}")
            else:
                # A wrapped continuation line (no label) is all description — dim it whole.
                lead = line[: len(line) - len(line.lstrip(" "))]
                out.append(lead + dim(line[len(lead) :]))
        else:
            out.append(line)
    return "\n".join(out)


def configure_help(root: click.Group) -> None:
    """Attach curated short help and examples, and upgrade the group class (typo suggestions +
    intent-grouped root listing)."""
    root.epilog = _root_epilog()
    if root.help:
        root.help = _dim_capabilities_block(root.help)
    for path, command in _walk(root):
        if path in _SHORT_HELP:
            command.short_help = _SHORT_HELP[path]
        examples = _EXAMPLES.get(path)
        if examples:
            command.epilog = _example_epilog(examples)
    apply_group_class(root, _COMMAND_SECTIONS)
