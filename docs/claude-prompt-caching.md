# Claude prompt caching in Agent Bricks templates

The LangGraph and OpenAI Agents templates, including both custom-agent templates, enable explicit
Claude prompt caching through the shared Databricks client. The main templates use AI Gateway;
custom templates use Foundation Model APIs. Other providers receive their existing request format.

Set `DATABRICKS_PROMPT_CACHING=off` to disable automatic placement. For local development, put it in
`.env`; for deployment, set the application environment, for example through `env` in `app.yaml`.
Accepted values are `auto` and `off`.

## Existing generated projects

Templates are copied by `agentbricks init`; upgrading the CLI does not rewrite an existing agent.
Upgrade the project dependencies and update its model/client construction:

- OpenAI Agents: require `databricks-openai>=0.18.0` and pass
  `prompt_caching=os.getenv("DATABRICKS_PROMPT_CACHING", "auto")` to `AsyncDatabricksOpenAI`.
- LangGraph: require `databricks-langchain>=0.22.0` and pass the same setting to `ChatDatabricks`.
- If the Claude endpoint name does not contain `claude`, also pass
  `prompt_cache_provider="anthropic"`. Do not force this provider when using a multi-provider picker.

Import `os` if needed, then rebuild the environment with `agentbricks dev --prepare-environment`.
Release the OpenAI package before the LangChain package, then release Agent Bricks with the updated
dependency minimums. To test an unreleased checkout, use the local/git source overrides described in
the [Agent Bricks contribution guide](../integrations/agentbricks/CONTRIBUTING.md).

## Cache policy and verification

The policy places a cache point on the final system/developer message and the final three user or
tool-result request boundaries. A parallel group of tool results counts as one boundary. System
instructions cache the preceding tool definitions too. Keeping earlier boundaries allows reuse
even when a tool batch exceeds Claude's prefix lookback window. At most four points are added.
Caller-provided cache controls take precedence and disable automatic placement for that request.
Annotations are added to request copies, so session transcripts and checkpoints remain unchanged.

Five-minute ephemeral caching is a reuse optimization: short prompts below the model's threshold
remain uncached, changed prefixes miss, and expired entries need another write. Cache writes cost
more than ordinary input; savings depend on reuse. See [Databricks caching documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#prompt-cache)
and [Anthropic caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

A cache write does not guarantee a hit on the very next gateway request. The live client checks
allow up to four requests with an unchanged system prefix and require an actual cache read.
Agent and graph checks retain at least two conversation turns and allow up to four turns to
observe a real read; they fail if no read occurs within that bound.

Inspect actual cache-read counts, not just the presence of cache-control fields:

- OpenAI: `usage.prompt_tokens_details.cached_tokens`; writes are `cache_write_tokens`.
- LangChain: `usage_metadata["input_token_details"]["cache_read"]`; writes are `cache_creation`.

Databricks already includes cached input in the aggregate input-token count. Never add cache reads
or writes again when calculating totals. Native Claude counters remain available on OpenAI usage.

Live regression tests are opt-in and make billable model requests. Run from the respective package
directory with `DATABRICKS_CONFIG_PROFILE` set:

```sh
# Shared sync/async client and OpenAI Agents, AI Gateway and Foundation Model APIs:
RUN_CLAUDE_PROMPT_CACHING_TESTS=1 uv run pytest tests/integration_tests/test_prompt_caching.py -v

# LangGraph, including streaming and a tool loop:
RUN_CLAUDE_PROMPT_CACHING_TESTS=1 uv run pytest tests/integration_tests/test_chat_models.py -k automatic_prompt_caching -v
```

Override `CLAUDE_CACHE_GATEWAY_MODEL` or `CLAUDE_CACHE_FMAPI_MODEL` for workspaces exposing different
Claude models. The defaults are `system.ai.claude-sonnet-4-5` and `databricks-claude-sonnet-4-6`.
