# Framework Recovery With SSE Replay

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=True, sse_replay=True)
```

The server persists stream events for both recovery prose and client replay.
After a crash it rotates the SDK session and resumes from the persisted event
log.
