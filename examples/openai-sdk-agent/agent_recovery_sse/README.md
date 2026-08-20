# Agent Recovery With SSE Replay

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=False, sse_replay=True)
```

The agent SDK session store restores the transcript. The server retains stream
events only for client replay and resumes the same SDK session with a fixed
recovery prompt.
