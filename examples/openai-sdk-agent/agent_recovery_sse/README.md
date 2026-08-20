# Agent Recovery With SSE Replay

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=False, sse_replay=True)
```

The agent SDK session store restores the transcript. The server retains stream
events only for client replay and resumes the same SDK session with a fixed
recovery prompt.

- [`handlers.py`](./handlers.py) detects that fixed recovery prompt and invokes
  the resume path against the same SDK session. It also shows the optional
  `@on_resume()` callback in a commented block.
- [`app.py`](./app.py) directly constructs the server with
  `auto_recovery=False` and `sse_replay=True`.

## Physical persistence

| Logical store | Physical table | What this app writes |
| --- | --- | --- |
| Runtime durability | `agent_server.responses` | Original request, terminal response, status, heartbeat, attempt, handler mode |
| Durable event log | `agent_server.messages` | SSE events/output items for client replay; not used to rebuild agent context |
| SDK session identity | `openai_agent_sse_sessions.agent_sessions` | The stable SDK session ID used before and after recovery |
| SDK transcript | `openai_agent_sse_sessions.agent_messages` | The OpenAI SDK's user/model/tool transcript used for recovery |

The SDK schema is created by `AsyncDatabricksSession`; the runtime creates only
the `agent_server` schema.

### Captured streaming response

Captured from the deployed `durable-session-sse` branch on 2026-08-20:

| Runtime row | Value |
| --- | --- |
| `response_id` | `resp_d55a661e2da746ffaf4ac240` |
| `status / attempt_number / is_streaming` | `completed / 1 / true` |
| `original_request / response` | stored: `565 / 49,620` characters |
| terminal `output` items | `29` |

| Store | Persisted rows |
| --- | --- |
| Event log | `1,660` rows, sequences `0-1659` |
| SDK session transcript | `30` messages in the same session ID |

On a crash, the event rows remain available for client replay, while the fixed
recovery request reopens this same SDK session to restore agent context.
