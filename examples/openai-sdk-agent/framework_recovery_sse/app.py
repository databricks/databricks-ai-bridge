"""Framework-managed recovery with durable SSE replay."""

from shared.server_factory import create_server, run_server

agent_server = create_server(auto_recovery=True, sse_replay=True)
app = agent_server.app


if __name__ == "__main__":
    run_server("framework_recovery_sse.app:app")
