"""Shared fixtures for the Agent Bricks CLI unit tests."""

from __future__ import annotations

from unittest import mock

import pytest

from databricks_agentbricks.cli import deploy as deploy_mod


@pytest.fixture(autouse=True)
def _hermetic_trace_reconcile(monkeypatch):
    """Keep the deploy-time trace-resource reconcile hermetic for every deploy-command test.

    `agentbricks deploy` reconciles the agentbricks-owned trace app-resources on EVERY deploy (so an
    `agentbricks tracing unbind` + redeploy prunes stale grants). That reconcile shells out to the
    `databricks` CLI via `app_resources` - NOT the `deploy_mod._databricks` these tests patch - so an un-stubbed
    deploy-command test would invoke the real CLI, which isn't present in CI (uncaught
    ``FileNotFoundError`` -> the deploy command fails). Stubbing it here (autouse, dir-wide) keeps every
    deploy-command test module hermetic without each one re-stubbing it. Tests that assert on the
    reconcile override this with their own mock; `app_resources_test.py` exercises the real function
    directly (it imports it from `app_resources`, so this patch of `deploy_mod`'s reference is inert
    there).
    """
    monkeypatch.setattr(deploy_mod, "apply_trace_resources", mock.Mock(return_value=None))
