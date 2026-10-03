# Bring an existing agent

From the existing project, choose the command for your agent's framework to prepare
the migration for your coding agent.

LangGraph:

```sh
agentbricks init --framework langgraph --existing .
```

OpenAI Agents SDK:

```sh
agentbricks init --framework openai --existing .
```

Before or after the conversion, inspect its progress without changing the repository or contacting
Databricks:

```sh
agentbricks doctor .
agentbricks -o json doctor .
```

Doctor exits 0 only when the project has a valid Agent Bricks manifest and matching project
metadata, uses the Agent Bricks server, declares the framework-appropriate `databricks-agentbricks`
extra and a non-empty `app.yaml` command, constructs `DurableAgentServer` with an `invoke` hook, and
calls a recognized adapter for the selected framework in production Python source. Test, example, and
old/stale directories do not count as source evidence. A failed report is the normal result for a
project that still needs migration; run
`agentbricks init --framework <framework_name> --existing <directory>` with the appropriate framework
to prepare the migration instructions. Doctor never imports or executes the target's source, and a
bounded source scan that exceeds a limit is reported while the evidence it already found still counts.
Its findings are static repository evidence, not proof that the configured startup command executes
the files it finds.

This writes `agent-bricks-migrate/` containing a skill, a prompt to paste into your coding agent,
`references/migration.json`, and a reference project generated from the templates bundled with the
installed CLI. The bundle sits outside any single agent's configuration directory; `.claude/skills/`
and `.agent/skills/` each receive a small skill that points at it, so Claude Code, Codex, and
similar tools discover the same instructions without duplicating the reference. Agent Bricks CLI
prepares the instructions; the coding agent performs and verifies the conversion. Init leaves application
source, dependencies, `.env`, and existing `.agentbricks/project.toml` configuration intact and refuses to overwrite
existing migration files.

The bundle is scaffolding for the migration, not part of the application: delete `agent-bricks-migrate/`
and the two pointer skills once the conversion is done, and keep them out of commits meanwhile.

The skill follows the shared managed-runtime contract for the selected framework, included in new projects and
migration references:
[LangGraph](../src/databricks_agentbricks/templates/agent-langgraph/AGENTKIT_CONTRACT.md) or
[OpenAI Agents SDK](../src/databricks_agentbricks/templates/agent-openai/AGENTKIT_CONTRACT.md). It explicitly
handles existing history, custom state and output, recovery, and client/session contracts. For
LangGraph, switching checkpointers does not migrate old conversations (likewise, the OpenAI Agents
SDK keeps prior Session transcripts and RunState behind); unresolved transitions require a user
decision.

The reference honors `--disable-chat-app`, `--memory-store`, `--session-store`, and the selected
profile. These are migration intent; init does not provision resources or change the existing
application. Migration supports LangGraph and the OpenAI Agents SDK with the managed server (`server = "agentbricks"`);
`--server custom` is not supported for `--existing`.
