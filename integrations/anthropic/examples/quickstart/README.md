# Quickstart

One Claude Managed Agents session whose tools run in a Databricks Sandbox.

```bash
uv run examples/quickstart/main.py --profile <databricks-profile>
# browser login instead of the CLI:
uv run examples/quickstart/main.py --auth browser --host https://<workspace>.cloud.databricks.com
```

What happens:

1. You log in to Databricks (U2M); the sandbox will act as you.
2. An agent with the built-in toolset is created. Pass `--agent-id` next time to reuse it.
3. A session is created on your self-hosted environment, with a `--budget-cents` spend cap (default $2), and the task is sent.
4. The dispatcher claims the session, creates its sandbox (`cma-…`), installs the worker and starts it.
5. Claude's `bash` call runs inside the sandbox. The script then reads `hello.txt` back through the Databricks SDK.
6. The session and sandbox are deleted, unless you pass `--keep`.

Example output:

```
Databricks: you@example.com on https://<workspace>.cloud.databricks.com (databricks-cli)
Session sesn_016h…
  tool  : bash {'command': "cd /home/sandbox-agent/agent-workspace && { echo 'Hello from a Databricks Sandbox!'; uname -a; } > hello.txt && cat hello.txt"}
Worker running in sandboxes/cma-3285ab3488f680900385bf5d (8.5s to launch)
  claude: I wrote the greeting and the `uname -a` output to hello.txt in the agent workspace.
/home/sandbox-agent/agent-workspace/hello.txt, read from Databricks:
Hello from a Databricks Sandbox!
Linux databricks 6.18.58 … x86_64 GNU/Linux
Deleted the session and its sandbox.
```

Options: `--env-file` (default `.env.anthropic`), `--model` (default `claude-opus-5-5`), `--prompt`, `--auth cli|browser|auto`,
`--oauth-env-file` (your OAuth app's `DATABRICKS_U2M_*`), `--keep`.
