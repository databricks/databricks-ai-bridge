# Cookbooks

These focused cookbooks show how to wire `databricks-ai-bridge` APIs into runnable applications.

| Cookbook | Agent state | Recovery strategies |
| --- | --- | --- |
| [OpenAI Agents SDK](./openai-sdk-agent/README.md) | `AsyncDatabricksSession` transcript | Event log and agent session |
| [LangGraph](./langgraph-agent/README.md) | `AsyncCheckpointSaver` graph checkpoints | Event log and native checkpoint resume |

Both cookbooks use the same long-running wait tool so their crash-recovery
behavior is easy to compare.
