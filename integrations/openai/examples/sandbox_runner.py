"""Minimal Databricks-backed sandbox example for the OpenAI Agents SDK.

This mirrors the hosted sandbox extension examples in the Agents SDK repo: it
creates a tiny workspace, asks a sandboxed agent to inspect it through one shell
tool, and prints a short answer. The shell command runs inside a Databricks
sandbox provisioned by ``DatabricksSandboxClient``.

Requirements:

1. ``pip install databricks-openai[sandbox]``.
2. A sandbox-capable Databricks workspace and authentication. Pass ``--profile``
   to select a configured CLI profile, or configure the SDK through the standard
   ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` environment variables.
3. Access to a Databricks model-serving endpoint (``--model``). The profile's
   token drives the agent's model turns through ``/serving-endpoints``; no OpenAI
   API key is required.

Run:

    python integrations/openai/examples/sandbox_runner.py --profile my-profile --stream
"""

from __future__ import annotations

import argparse
import asyncio

from agents import Runner, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.entries import File
from databricks.sdk import WorkspaceClient
from openai.types.responses import ResponseTextDeltaEvent

from databricks_openai import (
    AsyncDatabricksOpenAI,
    DatabricksSandboxClient,
    DatabricksSandboxClientOptions,
)

DEFAULT_MODEL = "databricks-claude-sonnet-5-5"
DEFAULT_QUESTION = "Summarize this sandbox workspace in two sentences, citing the file names."


def _build_manifest() -> Manifest:
    return Manifest(
        entries={
            "README.md": File(
                content=b"# Databricks Demo Workspace\n\n"
                b"This workspace validates the Databricks sandbox backend.\n"
            ),
            "tasks.md": File(
                content=b"# Tasks\n\n"
                b"1. Inspect the workspace files.\n"
                b"2. Summarize the setup in two sentences.\n"
            ),
        }
    )


def _serving_model(profile: str | None, endpoint: str) -> OpenAIChatCompletionsModel:
    """Build a chat-completions model backed by a Databricks serving endpoint.

    ``AsyncDatabricksOpenAI`` targets the workspace's ``/serving-endpoints`` and
    authenticates with a refreshing bearer token from the ``WorkspaceClient``, so
    it works with PAT, OAuth, and M2M profiles rather than a static API key.
    """
    client = AsyncDatabricksOpenAI(workspace_client=WorkspaceClient(profile=profile))
    return OpenAIChatCompletionsModel(model=endpoint, openai_client=client)


async def main(*, profile: str | None, model: str, question: str, stream: bool) -> None:
    set_tracing_disabled(True)

    agent = SandboxAgent(
        name="Databricks Sandbox Assistant",
        model=_serving_model(profile, model),
        instructions=(
            "Answer questions about the sandbox workspace. Inspect the files with the shell "
            "before answering, keep the response concise, and cite the file names you inspected. "
            "Do not invent files that are not present."
        ),
        default_manifest=_build_manifest(),
        capabilities=[Shell()],
    )

    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=DatabricksSandboxClient(profile=profile),
            options=DatabricksSandboxClientOptions(),
        ),
        workflow_name="Databricks sandbox example",
    )

    if not stream:
        result = await Runner.run(agent, question, run_config=run_config)
        print(result.final_output)
        return

    stream_result = Runner.run_streamed(agent, question, run_config=run_config)
    saw_text_delta = False
    async for event in stream_result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if not saw_text_delta:
                print("assistant> ", end="", flush=True)
                saw_text_delta = True
            print(event.data.delta, end="", flush=True)
    if saw_text_delta:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Databricks CLI profile to authenticate.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Databricks serving endpoint name.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Prompt to send to the agent.")
    parser.add_argument("--stream", action="store_true", default=False, help="Stream the response.")
    args = parser.parse_args()

    asyncio.run(
        main(profile=args.profile, model=args.model, question=args.question, stream=args.stream)
    )
