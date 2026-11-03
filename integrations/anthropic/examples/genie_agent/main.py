"""Ask Claude a data question; Genie One answers it from inside a Databricks Sandbox.

The dispatcher starts the session's worker with Genie One configured. When Claude calls
``genie_ask_and_wait``, the worker asks Genie One and polls it *inside the sandbox* with the
sandbox's built-in credential (your identity), so Unity Catalog permissions apply and no
Databricks token is ever given to Anthropic.

    uv run --extra mcp examples/genie_agent/main.py \
        --profile my-workspace --agent-id agent_... \
        --question "Which 5 pickup zip codes have the most trips in samples.nyctaxi.trips?"
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shared  # noqa: E402
import setup_agent  # noqa: E402

from databricks_anthropic import (  # noqa: E402
    DatabricksSandboxDispatcher,
    DatabricksSandboxLauncher,
)
from databricks_anthropic.sandbox.mcp import genie_one_server  # noqa: E402

DEFAULT_QUESTION = "Which 5 pickup zip codes have the most trips in samples.nyctaxi.trips?"


async def main(args: argparse.Namespace) -> int:
    settings = _shared.load_settings(args)
    w = _shared.databricks_login(args)
    api = _shared.anthropic_client(settings)
    agent_id = args.agent_id or await setup_agent.create_agent(args, settings=settings, w=w)

    launcher = DatabricksSandboxLauncher(
        workspace_client=w,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        environment_key=settings["ANTHROPIC_ENVIRONMENT_KEY"],
        mcp_servers=[genie_one_server(w)],  # worker serves Genie One tools from inside the sandbox
        max_idle_s=60,
    )
    session = await api.beta.sessions.create(
        agent=agent_id,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        title="Genie One in a Databricks Sandbox",
        budget=_shared.budget(args.budget_cents),
    )
    print(f"Session {session.id}")
    try:
        ready = asyncio.Event()
        turn = asyncio.create_task(_shared.stream_turn(api, session.id, ready))
        await asyncio.wait_for(ready.wait(), 30)
        await api.beta.sessions.events.send(
            session.id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": args.question}]}],
        )
        await asyncio.sleep(2)

        dispatcher = DatabricksSandboxDispatcher(launcher, drain=True)
        for result in await dispatcher.run():
            print(f"Worker running in {result.sandbox_name} ({result.seconds:.1f}s to launch)")
        for session_id, error in dispatcher.failures:
            print(f"Launch failed for {session_id}: {error}")
        answer = await turn
        return 0 if answer else 1
    finally:
        if args.keep:
            print(f"Kept session {session.id} and sandbox {launcher.sandbox_name_for(session.id)}")
        else:
            with contextlib.suppress(Exception):
                await api.beta.sessions.delete(session.id)
            launcher.delete_session_sandbox(session.id)
            print("Deleted the session and its sandbox.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _shared.add_args(parser)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    sys.exit(asyncio.run(main(parser.parse_args())))
