# Agent Recovery With Polling

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=False, sse_replay=False)
```

The agent SDK session store restores the transcript. The server writes no
stream-event rows and persists only durability metadata, the request, and the
terminal response used by polling.

- [`handlers.py`](./handlers.py) uses one normal invoke/stream path for initial
  and resumed requests. The commented `@on_resume()` block shows the complete
  request translation that happens before the same handler is called.
- [`app.py`](./app.py) directly constructs the server with both policy flags
  set to `False`.

## Physical persistence

| Logical store | Physical table | What this app writes |
| --- | --- | --- |
| Runtime durability | `agent_server.responses` | Original request, terminal response, status, heartbeat, attempt, handler mode |
| Durable event log | `agent_server.messages` | No rows; the table exists only for schema compatibility |
| SDK session identity | `openai_agent_polling_sessions.agent_sessions` | The stable SDK session ID used before and after recovery |
| SDK transcript | `openai_agent_polling_sessions.agent_messages` | The OpenAI SDK's user/model/tool transcript used for recovery |

The SDK schema is created by `AsyncDatabricksSession`; the runtime creates only
the `agent_server` schema.

The initial background request may provide `context.conversation_id`,
`custom_inputs.session_id`, or `custom_inputs.thread_id`. If omitted, the
server warns and injects `response_id` as `context.conversation_id` before the
first handler call, so the same generated SDK session can be reopened after a
crash.

### Captured polling response

Captured from the deployed `durable-session-poll` branch on 2026-08-20:

| Runtime row | Value |
| --- | --- |
| `response_id` | `resp_208374e2b2c64101af5ff3a4` |
| `status / attempt_number / is_streaming` | `completed / 1 / false` |
| `original_request / response` | stored: `563 / 1,910` characters |
| terminal `output` items | `1` |

| Store | Persisted rows |
| --- | --- |
| Event log | `0` rows |
| SDK session transcript | `24` messages in the same session ID |

Polling reads the terminal payload from `agent_server.responses.response`; it
does not reconstruct output from stream events.
