"""Helpers shared by the examples: settings, Databricks login, and the session event stream."""

from __future__ import annotations

import argparse
import asyncio
import os

import anthropic
from databricks.sdk import WorkspaceClient

from databricks_anthropic import auth as dbx_auth
from databricks_anthropic.envfile import read_env_file

ANTHROPIC_API_URL = "https://api.anthropic.com"


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        default=".env.anthropic",
        help="dotenv with ANTHROPIC_API_KEY / _ENVIRONMENT_ID / _ENVIRONMENT_KEY",
    )
    parser.add_argument(
        "--auth",
        choices=["cli", "browser", "auto"],
        default="cli",
        help="Databricks U2M login: CLI profile (default) or SDK browser login",
    )
    parser.add_argument("--profile", help="Databricks CLI profile (with --auth cli)")
    parser.add_argument("--host", help="workspace URL (with --auth browser)")
    parser.add_argument(
        "--oauth-env-file", help="dotenv with DATABRICKS_U2M_* for your own OAuth app (optional)"
    )
    parser.add_argument(
        "--model", default="claude-opus-5-5", help="model for a newly created agent"
    )
    parser.add_argument("--agent-id", help="reuse an agent instead of creating one")
    parser.add_argument(
        "--budget-cents", type=int, default=200, help="hard per-session spend cap (USD cents)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the session and sandbox for inspection"
    )


def load_settings(args: argparse.Namespace) -> dict[str, str]:
    # ANTHROPIC_* in the shell (e.g. Claude Code routed through a gateway) must not redirect
    # these calls: the env file is the only source.
    for key in [k for k in os.environ if k.startswith("ANTHROPIC_")]:
        os.environ.pop(key)
    if not os.path.exists(os.path.expanduser(args.env_file)):
        raise SystemExit(
            f"{args.env_file} not found. Copy .env.anthropic.template to {args.env_file} and fill it in."
        )
    settings = read_env_file(args.env_file)
    missing = [
        k
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_ENVIRONMENT_ID", "ANTHROPIC_ENVIRONMENT_KEY")
        if not settings.get(k) or "REPLACE_ME" in settings[k]
    ]
    if missing:
        raise SystemExit(
            f"{args.env_file} is missing {missing}. Copy .env.anthropic.template and fill it in."
        )
    for key, value in read_env_file(args.oauth_env_file).items():
        if key.startswith("DATABRICKS_U2M_"):
            os.environ.setdefault(key, value)
    return settings


def databricks_login(args: argparse.Namespace) -> WorkspaceClient:
    try:
        w = dbx_auth.login(auth=args.auth, profile=args.profile, host=args.host)
    except dbx_auth.AuthSetupError as e:
        raise SystemExit(f"Databricks login is not set up.\n\n{e}") from None
    print(f"Databricks: {w.current_user.me().user_name} on {w.config.host} ({w.config.auth_type})")
    return w


def anthropic_client(settings: dict[str, str]) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=settings["ANTHROPIC_API_KEY"],
        base_url=settings.get("ANTHROPIC_BASE_URL") or ANTHROPIC_API_URL,
    )


async def stream_turn(
    api: anthropic.AsyncAnthropic, session_id: str, ready: asyncio.Event, *, timeout_s: float = 600
) -> str:
    """Print the session's events until the agent finishes its turn; return its last message."""
    last = ""

    async def read() -> str:
        nonlocal last
        stream = await api.beta.sessions.events.stream(session_id)
        async with stream:
            ready.set()
            async for event in stream:
                if event.type == "agent.message":
                    last = "".join(getattr(b, "text", "") for b in (event.content or []))
                    print(f"  claude: {last}")
                elif event.type in ("agent.tool_use", "agent.custom_tool_use"):
                    detail = str(getattr(event, "input", ""))[:140]
                    print(f"  tool  : {event.name} {detail}")
                elif event.type == "session.status_idle":
                    reason = getattr(getattr(event, "stop_reason", None), "type", None)
                    if reason == "budget_reached":
                        print("  session reached its budget")
                        return last
                    # requires_action = waiting for our worker to attach and answer; keep streaming.
                    if reason != "requires_action" and last:
                        return last
        return last

    # asyncio.wait_for, not asyncio.timeout: Python 3.10 is supported.
    return await asyncio.wait_for(read(), timeout_s)


def budget(cents: int) -> dict:
    return {"type": "limit", "max_list_cost": {"amount": str(cents), "currency": "USD"}}
