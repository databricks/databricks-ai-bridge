"""Agent-session-managed recovery with polling and no event log."""

from shared.server_factory import create_server, run_server

agent_server = create_server(auto_recovery=False, sse_replay=False)
app = agent_server.app


if __name__ == "__main__":
    run_server("agent_recovery_polling.app:app")
