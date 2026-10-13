# Claude prompt caching in Agent Bricks templates

The four LangGraph and OpenAI Agents templates enable Claude caching through a shared
provider adapter in `databricks_agentkit.claude`. The generic Databricks OpenAI and LangChain
connectors forward explicit controls and report usage; they do not choose a caching policy.
The standard templates retain their Sonnet 4.5 model default. Custom templates retain their GPT default.

Set `DATABRICKS_CLAUDE_PROMPT_CACHING=off` to disable the template adapter. The default is `auto`.
For local development, set it in `.env`; for deployment, set the application environment,
for example through `env` in `app.yaml`. This setting affects Claude requests only. It does
not enable or disable caching managed by other providers or caller-supplied cache controls.

## Placement and endpoint compatibility

For the verified endpoint names `system.ai.claude-sonnet-4-6` and
`databricks-claude-sonnet-4-6`, the adapter supplies top-level
`cache_control={"type": "ephemeral"}`. Claude moves the conversation breakpoint automatically.
The adapter also marks the final system/developer message: a fixed point protects the stable
instructions and preceding tool definitions when a long tool batch exceeds the 20-block lookback.

Older Claude endpoints, including Sonnet 4.5, reject the top-level field. For other endpoint
names containing `claude`, the adapter marks the final system/developer message and the last
three user or completed tool-result boundaries, at most four points. A parallel batch of tool
results counts as one boundary. There is no failed-request fallback or extra discovery request.

Other models and unrecognized aliases receive their existing request format. Endpoint names are
an explicit compatibility choice, not authoritative provider metadata. For a custom alias,
configure controls yourself using the framework's existing `extra_body` or content blocks.
Any caller-supplied cache control disables template placement for that request. This preserves
manual placement and TTL choices. Annotations are added to copies; stored transcripts and
checkpoints remain unchanged. Only Chat Completions requests use this adapter.

Five-minute caching requires a sufficiently long, repeated prefix. Short prompts, changed
prefixes, and expired entries may miss. Cache writes cost more than ordinary input; savings
depend on reuse. See [Anthropic caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
and [Databricks caching documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#prompt-cache).

## Existing generated projects

`agentbricks init` copies template code. Upgrading the CLI does not rewrite an existing agent.
Upgrade `databricks-agentbricks` to the release containing this adapter (the development version
is `0.6.0.dev0`), `databricks-openai>=0.18.0`, and, for LangGraph, `databricks-langchain>=0.22.0`.

OpenAI Agents applications can construct the provider client explicitly:

```python
import os
from databricks_agentkit.claude import ClaudeCachingOpenAI

client = ClaudeCachingOpenAI(
    use_ai_gateway=True,
    cache_policy=os.getenv("DATABRICKS_CLAUDE_PROMPT_CACHING", "auto"),
)
# Pass client to OpenAIChatCompletionsModel or set_default_openai_client.
```

LangGraph templates call `prepare_claude_request` from their existing routed model subclass:

```python
from databricks_agentkit.claude import prepare_claude_request

class TemplateChatDatabricks(ChatDatabricks):
    def _prepare_inputs(self, *args, **kwargs):
        return prepare_claude_request(
            super()._prepare_inputs(*args, **kwargs),
            os.getenv("DATABRICKS_CLAUDE_PROMPT_CACHING", "auto"),
        )
```

Preserve your existing workspace routing methods when adopting this change. Rebuild with
`agentbricks dev --prepare-environment`. Release OpenAI, LangChain, then Agent Bricks; release
stamping updates template dependency minimums. To test an unreleased checkout, use the local
source overrides in the [contribution guide](../integrations/agentbricks/CONTRIBUTING.md).

## Verification

OpenAI cache reads are in `usage.prompt_tokens_details.cached_tokens`; writes are in
`cache_write_tokens`. Native Claude counters remain available. LangChain reports `cache_read`
and `cache_creation` in `usage_metadata["input_token_details"]`. They are already included in
input totals. Streaming keeps the response usage rather than adding overlapping snapshots.

Unit tests cover placement, manual controls, aliases, off, transcript preservation, client
copies, serialization, streaming, and template wiring. Gated live tests verify actual reads:

```bash
cd integrations/openai
RUN_CLAUDE_PROMPT_CACHING_TESTS=1 uv run pytest tests/integration_tests/test_prompt_caching.py -v

cd ../langchain
RUN_CLAUDE_PROMPT_CACHING_TESTS=1 uv run pytest tests/integration_tests/test_chat_models.py -k prompt_caching -v
```

Set `CLAUDE_CACHE_GATEWAY_MODEL` or `CLAUDE_CACHE_FMAPI_MODEL` for your workspace's endpoints.
The SDK live-test endpoints default to Sonnet 4.6. AgentKit tests cover both 4.5 and 4.6;
`CLAUDE_CACHE_ADAPTER_ROUTE=fmapi` runs those tests through Foundation Model APIs. Native Gateway acceptance and writes were
observed during exploration; TPM rate limits prevented proving repeat reads on that route.
The maintained tests use Gateway's MLflow-compatible route and Foundation Model APIs.
