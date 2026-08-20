# Framework Recovery With SSE Replay

This App uses the shared OpenAI Agents SDK implementation with:

```python
LongRunningAgentServer(auto_recovery=True, sse_replay=True)
```

The server persists stream events for both recovery prose and client replay.
After a crash it rotates the SDK session and resumes from the persisted event
log.

- [`handlers.py`](./handlers.py) always runs the normal review path and passes
  the framework-generated recovery prose into the new SDK session. It also
  shows the optional `@on_resume()` callback in a commented block.
- [`app.py`](./app.py) directly constructs the server with both policy flags
  set to `True`.

## Physical persistence

| Logical store | Physical table | What this app writes |
| --- | --- | --- |
| Runtime durability | `agent_server.responses` | Original request, terminal response, status, heartbeat, attempt, handler mode |
| Durable event log | `agent_server.messages` | Every persisted SSE event/output item, tagged by attempt and sequence |
| SDK session identity | `openai_framework_sse_sessions.agent_sessions` | Original and rotated `::attempt-N` session IDs |
| SDK transcript | `openai_framework_sse_sessions.agent_messages` | The OpenAI SDK's user/model/tool transcript for each session |

The SDK schema is created by `AsyncDatabricksSession`; the runtime creates only
the `agent_server` schema.

### Captured crash-recovery response

Captured from the deployed `durable-framework-sse` branch on 2026-08-20:

| Runtime row | Value |
| --- | --- |
| `response_id` | `resp_5bd6b74ead344cadb4aad5dc` |
| `status / attempt_number / is_streaming` | `completed / 2 / true` |
| `original_request / response` | stored: `566 / 58,214` characters |
| terminal `output` items | `21` |

| Attempt/session | Persisted rows |
| --- | --- |
| Event log, attempt 1 | `11` rows, sequences `0-10` |
| Event log, attempt 2 | `1,653` rows, sequences `11-1663` |
| Original SDK session | `1` message |
| Rotated SDK session `::attempt-2` | `22` messages |

This demonstrates that runtime recovery uses the event log and starts a new SDK
session; it does not repair or continue the crashed SDK transcript in place.
