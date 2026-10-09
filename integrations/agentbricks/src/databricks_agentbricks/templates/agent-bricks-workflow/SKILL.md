---
name: agent-bricks-workflow
description: Build, develop, deploy, validate, or troubleshoot custom agents with the Agent Bricks CLI and AgentKit. Use for agentbricks commands, agent.toml bindings, and Agent Bricks project integration; not for unrelated agent frameworks or MLflow-only tasks.
---

# Agent Bricks workflows

Use the installed `agentbricks` CLI and the project's declared framework, server, profile, and
bindings. Preserve the user's chosen models, tools, application behavior, and approval policy.
Skill discovery does not authorize provisioning, deployment, live calls, or deletion.

## Choose the task

- **New agent:** `agentbricks init <directory>` scaffolds a project and installs this workflow skill.
  LangGraph is the default; use `--framework openai` for an OpenAI Agents SDK project.
  Inspect `agent.toml` and the generated project instructions before changing integration points.
- **Existing agent adoption:** inspect its framework, then use
  `agentbricks init --existing --framework <framework> <directory>` when migration is requested.
  Read the generated `agent-bricks-migrate/SKILL.md` and its references. Migration remains a separate
  workflow because it must preserve existing state and client contracts. Do not scaffold over an
  existing application or migrate it merely to debug a CLI command.
- **Local development:** use `agentbricks dev` when local iteration is needed. Read
  [development and deployment](references/deployment.md) for local-versus-deployed behavior.
- **Deployment or proof of life:** read [development and deployment](references/deployment.md).
  If the goal is a deployed agent, a local dev server is optional, not a prerequisite. Retain
  relevant offline checks and perform only the live validation the user has requested.
- **Tools, memory, sessions, or tracing:** read [state and tools](references/state-and-tools.md)
  for resource identity, permissions, and focused diagnostics.

## Work from installed evidence

Run `agentbricks <command> --help` for the relevant command, not every command in the CLI.
Global `--profile/-p` and `--output/-o` precede subcommands. Use `--output json` when consuming
command results programmatically. Resource lists are workspace-wide, not proof of project ownership.

In scaffolded projects, `AGENTKIT_CONTRACT.md` owns runtime wiring and transport requirements;
`AGENTS.md` and `README.md` explain the project's code and setup. Read only references needed for the
task. Do not invent SDK APIs or replace integration points with a bespoke server.

`agentbricks skills show` reads the skill bundled with the installed CLI.
`agentbricks skills install <directory>` adopts that version in an existing project without
overwriting a different skill. A project-local copy can be older than the installed CLI; inspect
command help before relying on an old option. Some coding agents require a new session to discover
newly installed skills. Any agent can instead read this file directly.

Report what changed, what was validated locally versus in the workspace, and any unresolved
permissions or state decisions. A configured binding or successful deployment alone does not prove
that the agent's behavior works.
