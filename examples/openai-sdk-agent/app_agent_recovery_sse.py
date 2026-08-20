"""Agent-session-managed recovery with durable SSE replay."""

from server_factory import create_server, run_server

agent_server = create_server(auto_recovery=False, sse_replay=True)
app = agent_server.app


if __name__ == "__main__":
    run_server("app_agent_recovery_sse:app")
