# Databricks sandbox x OpenAI Agents API - interactive demo

Talk to an OpenAI Agents API agent that executes inside a Databricks sandbox. You type
an instruction; the agent decides which shell commands to run in the sandbox and streams
back the command, its output, and a reply. This is the application-managed integration:
your local process owns the whole lifecycle and makes every outbound call - OpenAI never
calls Databricks.

## What happens

1. Create an Agents API `self_hosted` session (OpenAI hosts the agent harness).
2. Provision a Databricks sandbox and wait until it is RUNNING.
3. Launch `codex exec-server` inside the sandbox; it opens one long-lived outbound
   connection to OpenAI for the session.
4. Drop into a REPL. Each line you type is one turn against the session. Tool calls
   (shell commands) run in the sandbox; output flows back through OpenAI to your terminal.
5. On exit, stop the executor, delete the session, and delete the sandbox. Pass
   `--keep-sandbox` to leave it (which keeps its egress IP stable for the next run).

## Prerequisites

- `uv` on PATH (resolves the script's inline dependencies - no venv needed).
- A Databricks workspace profile in `~/.databrickscfg` with sandbox access (the demo was
  validated on `personal-staging`).
- Two OpenAI keys from the same org/project:
  - application key - creates sessions. Default file `~/.oai_key_staging`, or `OPENAI_API_KEY`.
  - environment key - created on the OpenAI Agents tab (Environments > keys) with every
    other permission None; passed into the sandbox as `CODEX_API_KEY`. Default file
    `~/.oai_env_key`, or `OPENAI_EXECUTOR_API_KEY`. A general key fails the executor
    connect with `missing required scope api.agents.environments.connect`.
- If your OpenAI org enforces an IP allowlist, the sandbox's egress IP must be on it. The
  demo prints that IP during setup; if the executor can't connect, add it on the org.
  (Sandboxes in the same workspace typically share one egress IP, so this is often already
  covered.) The `--keep-sandbox` flag (see Options) persists a sandbox if you ever want a
  stable egress IP across runs, but the default deletes it.

## Run

    uv run interactive.py --profile personal-staging

Then, at the `you>` prompt, try:

    what OS and CPU is this sandbox running on?
    create a python venv, install pandas, and show me its version
    write a fib(30) script, run it, and tell me the result

The sandbox is authenticated to your Databricks workspace (ESM injects a caller-scoped
token on create), so the agent can query Unity Catalog and SQL warehouses as you - no
setup inside the sandbox.

### Slash commands

- `/demo` - run a canned analysis of `samples.nyctaxi` and save a chart (then `/pull` it)
- `/pull <path> [local-name]` - copy a file out of the sandbox to `./pulled/` (path relative to the work dir)
- `/sh <command>` - run a command directly in the sandbox (ground truth, bypasses the agent)
- `/info` - session id, sandbox name, model, work dir
- `/verbose` - toggle showing the agent's reasoning summary
- `/help` - command list
- `/quit` - end and tear down (Ctrl-D is the same). Ctrl-C cancels an in-flight turn.

## Options

| flag | default | meaning |
| --- | --- | --- |
| `--profile` | `$DATABRICKS_CONFIG_PROFILE` | Databricks config profile |
| `--sandbox-id` | `openai-agent-demo` | sandbox to create or reuse |
| `--model` | `gpt-5.6-sol` | Agents API model |
| `--inactivity-timeout` | `900` | sandbox auto-stop idle seconds |
| `--app-key-file` | `~/.oai_key_staging` | file with the application key |
| `--executor-key-file` | `~/.oai_env_key` | file with the environment key |
| `--keep-sandbox` | off (deleted) | keep the sandbox on exit instead of deleting it |
| `--verbose` | off | stream the agent's reasoning summary |

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `missing required scope api.agents.environments.connect` | executor key is a general key, not an Agents-tab environment key |
| `codex exec-server did not stay connected` | OpenAI org IP allowlist is missing the sandbox egress IP the demo printed; add it, or reuse `oai-agent-cookbook-test`, or disable the allowlist on a test org |
| session creation 401/403 | application key wrong org/project, or Agents API not enabled for the org |
| sandbox never reaches RUNNING | workspace profile lacks sandbox access, or region capacity |
| `turn error: An internal error occurred` | transient Agents API error. The REPL auto-recovers the session to idle; just resend your message. It does not auto-resend (a dropped stream may hide a turn that still ran), so you decide whether to repeat it. |
| `requires an idle session` | a prior turn is still running server-side; wait a moment (the prompt tells you) and resend |

## Files

- `interactive.py` - the demo (REPL + lifecycle). Run this.
- `common.py` - provisioning, executor launch, and sandbox-exec helpers, taken verbatim
  from the published cookbook example (`openai/openai-cookbook`,
  `examples/agents_api/sandboxes/databricks`).
