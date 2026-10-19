"""Live lifecycle tests for ``DatabricksSandboxClient`` / ``DatabricksSandboxSession``.

Provisions a real Databricks sandbox through ``DatabricksSandboxClient`` (the
client owns the sandbox lifecycle in the Agents SDK), exercises ``exec`` / ``read``
/ ``write`` / ``persist_workspace`` / ``hydrate_workspace`` through the public
``SandboxSession`` wrapper, and tears it down. No model serving is needed.

These tests require:
1. A sandbox-capable Databricks workspace and valid authentication (set
   ``DATABRICKS_SANDBOX_PROFILE`` to a configured CLI profile, or configure the
   SDK through the standard DATABRICKS_HOST/token environment variables).
2. ``databricks-openai[sandbox]`` installed.

Enable by setting ``RUN_SANDBOX_INTEGRATION_TESTS=1``:
    RUN_SANDBOX_INTEGRATION_TESTS=1 DATABRICKS_SANDBOX_PROFILE=my-profile \\
        pytest tests/integration_tests/test_openai_sandbox.py -v
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

pytest.importorskip("agents.sandbox.session.sandbox_client")
pytest.importorskip("databricks.sdk.service.sandbox")

from databricks_openai import (  # noqa: E402
    DatabricksSandboxClient,
    DatabricksSandboxClientOptions,
)

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RUN_SANDBOX_INTEGRATION_TESTS"),
        reason="RUN_SANDBOX_INTEGRATION_TESTS not set",
    ),
    pytest.mark.asyncio,
]


def _client() -> DatabricksSandboxClient:
    return DatabricksSandboxClient(profile=os.environ.get("DATABRICKS_SANDBOX_PROFILE"))


async def test_lifecycle_exec_and_file_round_trip() -> None:
    client = _client()
    session = await client.create(options=DatabricksSandboxClientOptions())
    try:
        await session.start()

        result = await session.exec("echo", "hello-sandbox", shell=False)
        assert result.ok()
        assert b"hello-sandbox" in result.stdout

        root = session.state.manifest.root
        await session.write(Path(f"{root}/note.txt"), io.BytesIO(b"persisted"))
        stream = await session.read(Path(f"{root}/note.txt"))
        assert stream.read() == b"persisted"
    finally:
        await session.aclose()
        await client.delete(session)


async def test_persist_and_hydrate_workspace() -> None:
    payload = bytes(range(256)) * 1024  # 256 KiB, exercises multi-chunk base64 transfer
    client = _client()
    session = await client.create(options=DatabricksSandboxClientOptions())
    try:
        await session.start()
        root = session.state.manifest.root
        await session.write(Path(f"{root}/data.bin"), io.BytesIO(payload))

        archive = await session.persist_workspace()
        await session.exec("rm", "-f", f"{root}/data.bin", shell=False)
        await session.hydrate_workspace(archive)

        stream = await session.read(Path(f"{root}/data.bin"))
        assert stream.read() == payload
    finally:
        await session.aclose()
        await client.delete(session)
