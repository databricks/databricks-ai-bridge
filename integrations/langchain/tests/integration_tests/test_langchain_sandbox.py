"""Live conformance tests for ``DatabricksSandbox``.

Subclasses the official ``langchain-tests`` ``SandboxIntegrationTests`` suite --
the same contract check the other Deep Agents sandbox providers run -- and
supplies a ``sandbox`` fixture that provisions a real Databricks sandbox and
tears it down after the class. No model serving is needed.

These tests require:
1. A sandbox-capable Databricks workspace and valid authentication (set
   ``DATABRICKS_SANDBOX_PROFILE`` to a configured CLI profile, or configure the
   SDK through the standard DATABRICKS_HOST/token environment variables).
2. ``databricks-langchain[deepagents]`` and ``langchain-tests`` installed.

Enable by setting ``RUN_SANDBOX_INTEGRATION_TESTS=1``:
    RUN_SANDBOX_INTEGRATION_TESTS=1 DATABRICKS_SANDBOX_PROFILE=my-profile \
        pytest tests/integration_tests/test_langchain_sandbox.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("deepagents")
pytest.importorskip(
    "langchain_tests", reason="langchain-tests not installed (sandbox conformance suite)"
)

from databricks.sdk import WorkspaceClient  # noqa: E402
from langchain_tests.integration_tests import SandboxIntegrationTests  # noqa: E402

from databricks_langchain import DatabricksSandbox  # noqa: E402
from integration_tests._sandbox_provisioning import (  # noqa: E402
    delete_sandbox,
    provision_sandbox,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SANDBOX_INTEGRATION_TESTS"),
    reason="RUN_SANDBOX_INTEGRATION_TESTS not set",
)


class TestDatabricksSandbox(SandboxIntegrationTests):
    @property
    def sandbox_root_dir(self) -> str:
        # /home/sandbox-agent is the writable, restart-surviving volume in a
        # Databricks sandbox; "/" and "/tmp" are an ephemeral overlay.
        return "/home/sandbox-agent/test_sandbox_ops/"

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        client = WorkspaceClient(profile=os.environ.get("DATABRICKS_SANDBOX_PROFILE"))
        name = provision_sandbox(client, f"langchain-it-{uuid.uuid4().hex[:8]}")
        try:
            yield DatabricksSandbox(client, name)
        finally:
            delete_sandbox(client, name)
