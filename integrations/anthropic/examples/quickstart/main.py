"""Quickstart: one Claude Managed Agents session whose tools run in a Databricks Sandbox.

1. Log in to Databricks (U2M); the sandbox will act as you.
2. Create (or reuse) an agent with the built-in agent toolset.
3. Create a session on your self-hosted environment and send a task.
4. Run the dispatcher until the queue is drained: it creates the session's sandbox and
   starts the Claude worker inside it.
5. Stream the agent's work, then read the file it wrote, from the Databricks side.

    uv run examples/quickstart/main.py --profile my-workspace
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _shared  # noqa: E402

from databricks_anthropic import (  # noqa: E402
    DatabricksSandboxDispatcher,
    DatabricksSandboxLauncher,
)

WORKDIR = "/home/sandbox-agent/agent-workspace"
DEFAULT_PROMPT = (
    f"Write 'Hello from a Databricks Sandbox!' and the output of `uname -a` to {WORKDIR}/hello.txt, "
    "then tell me what you wrote."
)


async def main(args: argparse.Namespace) -> int:
    settings = _shared.load_settings(args)
    w = _shared.databricks_login(args)
    api = _shared.anthropic_client(settings)

    agent_id = args.agent_id
    if not agent_id:
        agent = await api.beta.agents.create(
            name="databricks-sandbox-quickstart",
            model=args.model,
            system=f"You run inside a Databricks Sandbox. Your working directory is {WORKDIR}.",
            tools=[{"type": "agent_toolset_20260401"}],
        )
        agent_id = agent.id
        print(f"Created agent {agent_id}; pass --agent-id {agent_id} to reuse it.")

    launcher = DatabricksSandboxLauncher(
        workspace_client=w,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        environment_key=settings["ANTHROPIC_ENVIRONMENT_KEY"],
        max_idle_s=60,  # stop the sandbox one minute after the session goes quiet
    )
    session = await api.beta.sessions.create(
        agent=agent_id,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        title="Databricks Sandbox quickstart",
        budget=_shared.budget(args.budget_cents),
    )
    print(f"Session {session.id}")
    sandbox = launcher.sandbox_name_for(session.id)
    try:
        ready = asyncio.Event()
        turn = asyncio.create_task(_shared.stream_turn(api, session.id, ready))
        await asyncio.wait_for(ready.wait(), 30)
        await api.beta.sessions.events.send(
            session.id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": args.prompt}]}],
        )
        await asyncio.sleep(2)  # let the session enqueue its work item

        dispatcher = DatabricksSandboxDispatcher(launcher, drain=True)
        for result in await dispatcher.run():
            print(f"Worker running in {result.sandbox_name} ({result.seconds:.1f}s to launch)")
        for session_id, error in dispatcher.failures:
            print(f"Launch failed for {session_id}: {error}")
        await turn

        # Read the file back through the Databricks SDK, from outside the sandbox.
        result = w.sandbox.execute_command_sync(
            sandbox, "/bin/bash", args=["-lc", f"cat {WORKDIR}/hello.txt"]
        )
        content = result.stdout or ""
        print(f"\n{WORKDIR}/hello.txt, read from Databricks:\n{content.strip() or '(not found)'}")
        return 0 if content else 1
    finally:
        if args.keep:
            print(f"Kept session {session.id} and sandbox {sandbox}")
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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    sys.exit(asyncio.run(main(parser.parse_args())))
