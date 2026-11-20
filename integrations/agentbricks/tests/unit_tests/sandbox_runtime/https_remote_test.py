import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.sandbox.lakebox import HttpsRemote, RemoteError


def completed(**changes):
    return {
        "status": "EXECUTE_COMMAND_STATUS_COMPLETED",
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "truncated": False,
        **changes,
    }


def test_stdin_transfer_is_binary_exact_across_chunks_and_cleans_up():
    workspace = MagicMock()
    writes = []

    def execute(method, path, *, body, headers):
        assert method == "POST" and path.endswith("/sandboxes/ours/exec-sync")
        assert all(len(arg) < 128 * 1024 for arg in body["args"])
        if body["cmd"] == "python3" and "os.open" in body["args"][1]:
            writes.append(body["args"])
        output = subprocess.run([body["cmd"], *body["args"]], capture_output=True, check=False)
        return completed(
            exit_code=output.returncode,
            stdout=output.stdout.decode(),
            stderr=output.stderr.decode(),
        )

    workspace.api_client.do.side_effect = execute
    data = bytes(range(256)) * 1800
    command = "python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'"
    remote = HttpsRemote(workspace, "ours")
    assert remote.run(command, data).strip() == hashlib.sha256(data).hexdigest()
    assert len(writes) == 3
    assert not Path(writes[0][2]).exists()


@pytest.mark.parametrize(
    "response,error",
    [
        (completed(exit_code=2, stderr="build failed"), RemoteError),
        (completed(truncated=True), RemoteError),
        (completed(status="EXECUTE_COMMAND_STATUS_TIMED_OUT"), TimeoutError),
    ],
)
def test_rejects_failed_or_incomplete_results(response, error):
    workspace = MagicMock()
    workspace.api_client.do.return_value = response
    with pytest.raises(error):
        HttpsRemote(workspace, "ours").run("true")


def test_idle_policy_is_applied_once_per_client_and_failed_patch_can_retry(tmp_path):
    from databricks_agentkit.sandbox.lakebox import Lakebox

    lakebox = Lakebox(MagicMock(), tmp_path, "test", transport="https", idle_timeout=1800)
    lakebox.api = MagicMock(side_effect=[RuntimeError("unavailable"), {}, {}])
    with pytest.raises(RuntimeError):
        lakebox.set_idle_timeout("ours")
    lakebox.set_idle_timeout("ours")
    lakebox.set_idle_timeout("ours")
    assert lakebox.api.call_count == 2
    assert lakebox.api.call_args.args[2] == {
        "sandbox_id": "ours",
        "idle_timeout": "1800s",
        "no_autostop": False,
    }
    lakebox.set_idle_timeout("another")
    assert lakebox.api.call_count == 3
