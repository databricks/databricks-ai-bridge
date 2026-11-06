# Genie One agent

Claude answers data questions with Databricks Genie One. The Genie calls run **inside the session's Databricks
Sandbox**, using the sandbox's credential for you, so Unity Catalog permissions apply and Anthropic never gets a
Databricks token.

```bash
# once: create the agent (reads Genie One's tools with your login; needs EXECUTE on system.ai.genie_one_mcp)
uv run --extra mcp examples/genie_agent/setup_agent.py --profile <databricks-profile>

# ask a question
uv run --extra mcp examples/genie_agent/main.py --profile <databricks-profile> --agent-id agent_... \
  --question "Which 5 pickup zip codes have the most trips in samples.nyctaxi.trips?"
```

Without `--agent-id`, `main.py` runs the setup step first.

## How the Genie tools work

| The agent sees | What runs in the sandbox |
|----------------|--------------------------|
| `genie_ask_and_wait(question, conversation_id?)` | The worker calls Genie One `genie_ask`, then polls `genie_poll_response` every 3 s until Genie finishes. It returns `status`, `final_answer`, `deep_link`, `conversation_id` and the SQL |
| `genie_get_query_result(conversation_id, response_id, item_id)` | Passed straight through to Genie One, for full result rows |

The worker connects to `https://<workspace>/ai-gateway/mcp-services/system.ai.genie_one_mcp` with the sandbox's
built-in credential, refreshed per request because it rotates hourly.

Measured on "get me total revenue from my bakehouse" (Claude Haiku 5.5):

| Genie tools exposed | Tool calls by Claude | Input tokens (cached reads) | Output tokens |
|---------------------|----------------------|-----------------------------|---------------|
| raw `genie_ask` + `genie_poll_response` | 17 | ~190K | ~2.2K |
| `genie_ask_and_wait` (this example) | 1 (7 polls inside the sandbox) | ~6–12K | ~110 |

Example output:

```
Genie One tools: ['genie_ask_and_wait', 'genie_get_query_result']
Session sesn_01Bd…
  tool  : genie_ask_and_wait {'question': 'What is the total revenue from the bakehouse?'}
Worker running in sandboxes/cma-c72f8e296dd272a6181fc691 (11.3s to launch)
  claude: Total bakehouse revenue is **$66,471** from 3,333 transactions, May 1–17, 2024.
[Explore in Databricks](https://<workspace>/one/chat/threads/…)
Deleted the session and its sandbox.
```

To expose Genie's raw tools instead, use `genie_one_server(w, wait_tool=False)` in both `setup_agent.py` and `main.py`.
