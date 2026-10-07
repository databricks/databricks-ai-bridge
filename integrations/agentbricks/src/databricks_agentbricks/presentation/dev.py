"""Human-facing output for the local development command."""

from __future__ import annotations

import pathlib

from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.endpoint import print_agent_invoke_command
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.services.dev.dev_service import DevPreview


def local_trace_label(preview: DevPreview) -> str | None:
    """Describe the local MLflow UI without requiring a stable experiment id."""
    if not preview.tracing_uri:
        return None
    if preview.local_experiment_name:
        return f"{preview.tracing_uri} (experiment name: {preview.local_experiment_name})"
    return preview.tracing_uri


def announce_bound_resources(preview: DevPreview) -> None:
    """Explain why workspace-bound resources are not used by a local run."""
    if preview.memory_store:
        render.console().print(
            f"[dim]Memory store '{preview.memory_store}' is bound but `agentbricks dev` runs with "
            "long-term memory off. Run `agentbricks deploy` to use bound store.[/]"
        )
    if preview.session_store:
        render.console().print(
            f"[dim]Session store '{preview.session_store}' is bound but `agentbricks dev` keeps "
            "conversation history in-process (not durable). Run `agentbricks deploy` to use bound store.[/]"
        )
    if preview.trace_experiment:
        render.console().print(
            f"[dim]Tracing experiment '{preview.trace_experiment}' is bound but `agentbricks dev` "
            "traces to a local MLflow server. Run `agentbricks deploy` to trace to the bound experiment.[/]"
        )


def announce_local_url(
    source_dir: pathlib.Path,
    port: int,
    server: AgentServer | None,
    trace_url: str | None = None,
    *,
    has_chat_ui: bool | None = None,
) -> None:
    """Show the chat URL or an invocation example before run-local blocks."""
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    if has_chat_ui is None:
        has_chat_ui = (source_dir / "runtime" / "ui.py").is_file()
    tool_step: str | tuple[str, str] = (
        "Edit agent/agent.py to give the agent a tool"
        if server == AgentServer.CUSTOM
        else ("agentbricks tools add mcp <service>", "Give the agent a tool")
    )
    if has_chat_ui:
        fields = {"Chat UI": base}
        if trace_url:
            fields["Traces"] = trace_url
        render.success(
            "Starting agent",
            fields=fields,
            next_steps=[
                f"Open {base} to chat with your agent",
                tool_step,
                ("agentbricks memory bind <store>", "Attach a memory / session store"),
                (f"agentbricks deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )
    else:
        uses_runtime_api = server == AgentServer.AGENTBRICKS
        endpoint = f"{base}/api/invocations" if uses_runtime_api else f"{base}/invocations"
        body = (
            '{"id": "00000000-0000-4000-8000-000000000000", '
            '"input": [{"role": "user", "content": "hi"}]}'
            if uses_runtime_api
            else '{"input": [{"role": "user", "content": "hi"}]}'
        )
        sample = f"curl -X POST {endpoint} -H 'Content-Type: application/json' -d '{body}'"
        fields = {"Invoke": f"POST {endpoint}"}
        if trace_url:
            fields["Traces"] = trace_url
        render.success(
            "Starting API-only agent (no chat UI — see `agentbricks init --help`)",
            fields=fields,
            next_steps=[
                (sample, "Send a test request"),
                tool_step,
                (f"agentbricks deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )

    print_agent_invoke_command(
        f"--url {base}", uses_runtime_api=has_chat_ui or server == AgentServer.AGENTBRICKS
    )
