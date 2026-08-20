# Agent Recovery With Polling

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=False, sse_replay=False)
```

The agent SDK session store restores the transcript. The server writes no
stream-event rows and persists only durability metadata, the request, and the
terminal response used by polling.
