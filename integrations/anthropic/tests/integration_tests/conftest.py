"""Live tests for Claude Managed Agents on Databricks Sandbox.

Configuration (environment variables):

  RUN_SANDBOX_INTEGRATION_TESTS=1  enable the prerequisite check (creates one short-lived sandbox; no model calls)
  RUN_SANDBOX_E2E=1          also enable the end-to-end tests (real Claude sessions; a few cents on Haiku)

  DATABRICKS_SANDBOX_PROFILE Databricks CLI profile (default: the SDK's default auth, e.g. DATABRICKS_HOST +
                             DATABRICKS_CLIENT_ID/SECRET for a service principal)
  ANTHROPIC_ENV_FILE         dotenv with ANTHROPIC_API_KEY / _ENVIRONMENT_ID / _ENVIRONMENT_KEY
                             (default: .env.anthropic)
  SANDBOX_E2E_MODEL          agent model (default claude-haiku-5-5: test-only, cheapest current model)
  SANDBOX_E2E_BUDGET_CENTS   hard per-session cap at list price (default 100 = $1)
  SKIP_GENIE=1               skip Genie One checks (prerequisite + Genie end-to-end)

Test-only cost settings: the cheapest model, prompts that demand replies under 25 words, single-step tasks,
reused agents (found by metadata), and a hard per-session budget.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import anthropic
import pytest
from _support import ANTHROPIC_API_URL, SessionRun, read_env_file, run_session_async
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ambient ANTHROPIC_* (e.g. Claude Code routed through a Databricks gateway) must not redirect test calls.
for _key in [k for k in os.environ if k.startswith("ANTHROPIC_") and k != "ANTHROPIC_ENV_FILE"]:
    os.environ.pop(_key)


HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    # This hook sees every collected test; gate only the live tests in this folder.
    for item in [i for i in items if HERE in Path(str(i.path)).resolve().parents]:
        if "e2e" in item.keywords and not os.environ.get("RUN_SANDBOX_E2E"):
            item.add_marker(
                pytest.mark.skip(reason="set RUN_SANDBOX_E2E=1 (runs real Claude sessions)")
            )
        if not os.environ.get("RUN_SANDBOX_INTEGRATION_TESTS"):
            item.add_marker(
                pytest.mark.skip(
                    reason="set RUN_SANDBOX_INTEGRATION_TESTS=1 (uses your Databricks workspace)"
                )
            )
        if "genie" in item.keywords and os.environ.get("SKIP_GENIE"):
            item.add_marker(pytest.mark.skip(reason="SKIP_GENIE is set"))


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: runs a real Claude Managed Agents session")
    config.addinivalue_line(
        "markers", "genie: needs Genie One (EXECUTE on system.ai.genie_one_mcp)"
    )


# ------------------------------------------------------------------ settings and clients


@pytest.fixture(scope="session")
def settings() -> dict[str, str]:
    path = os.path.expanduser(
        os.environ.get("ANTHROPIC_ENV_FILE", str(REPO_ROOT / ".env.anthropic"))
    )
    if not os.path.exists(path):
        pytest.fail(
            f"{path} not found: copy .env.anthropic.template and fill it in, or set ANTHROPIC_ENV_FILE"
        )
    values = read_env_file(path)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_ENVIRONMENT_ID", "ANTHROPIC_ENVIRONMENT_KEY"):
        if not values.get(key) or "REPLACE_ME" in values[key]:
            pytest.fail(f"{key} is missing in {path}")
    return values


@pytest.fixture(scope="session")
def workspace_client() -> WorkspaceClient:
    # Config first, then WorkspaceClient(config=...), so the client carries the connector's User-Agent product.
    try:
        w = WorkspaceClient(config=Config(profile=os.environ.get("DATABRICKS_SANDBOX_PROFILE")))
        w.current_user.me()
    except Exception as e:  # noqa: BLE001 - surface the login problem as a test failure
        pytest.fail(
            f"Databricks login is not set up ({type(e).__name__}: {e}). Run `databricks auth login "
            "--profile <name>` and set DATABRICKS_SANDBOX_PROFILE."
        )
    return w


@pytest.fixture(scope="session")
def api(settings) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=settings["ANTHROPIC_API_KEY"],
        base_url=settings.get("ANTHROPIC_BASE_URL") or ANTHROPIC_API_URL,
    )


@pytest.fixture
def run_session(settings):
    """Run one prompt through a real session; always deletes the session and its sandbox."""
    launchers = []

    def _run(
        launcher, agent_id: str, prompt: str, *, timeout_s: float = 420, while_open=None
    ) -> SessionRun:
        launchers.append(launcher)
        result = asyncio.run(
            run_session_async(
                settings, launcher, agent_id, prompt, timeout_s=timeout_s, while_open=while_open
            )
        )
        launchers[-1] = (launcher, result.session_id)
        return result

    yield _run
    for entry in launchers:
        if isinstance(entry, tuple):
            launcher, session_id = entry
            launcher.delete_session_sandbox(session_id)
