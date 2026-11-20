import asyncio
import json
import subprocess
import sys
import time
from types import ModuleType
from unittest.mock import ANY

import pytest

from databricks_agentkit.sandbox.runner import CALL_SCRIPT, SCRIPT
from databricks_agentkit.sandbox.streaming import StreamDecoder


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_runner_calls_agent_and_keeps_logs_out_of_stream(tmp_path, asynchronous, streaming):
    (tmp_path / "custom_agent.py").write_text(
        ("async " if asynchronous else "") + "def invoke(request, state_dir, *, emit):\n"
        '    print("agent log")\n'
        '    emit("Hello 🌍\\n")\n'
        '    return {"reply": request["input"]["message"], "state": str(state_dir)}\n'
    )
    path = tmp_path / "streams" / "attempt.jsonl"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "state"), "custom_agent:invoke"]
        + (["--stream-file", str(path)] if streaming else []),
        input=json.dumps({"input": {"message": "hello"}}),
        text=True,
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1])["reply"] == "hello"
    if streaming:
        assert path.stat().st_mode & 0o777 == 0o600
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"type": "delta", "delta": "Hello 🌍\n", "at": ANY},
            {"type": "end", "status": "completed"},
        ]


def test_runner_flushes_deltas_before_agent_returns_and_records_failure(tmp_path):
    (tmp_path / "custom_agent.py").write_text(
        "import time\n"
        "def invoke(request, state_dir, *, emit):\n"
        '    emit("Hello 🌍\\n")\n'
        "    time.sleep(.4)\n"
        '    raise ValueError("test agent failure")\n'
    )
    path = tmp_path / "attempt.jsonl"
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            SCRIPT,
            str(tmp_path),
            "custom_agent:invoke",
            "--stream-file",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
    ) as process:
        process.stdin.write(b"{}")
        process.stdin.close()
        deadline = time.monotonic() + 3
        while not path.exists() or not path.read_bytes():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert process.poll() is None
        assert json.loads(path.read_text().splitlines()[0])["delta"] == "Hello 🌍\n"
        assert process.wait(timeout=5) != 0
    assert json.loads(path.read_text().splitlines()[-1]) == {"type": "end", "status": "failed"}
    assert "test agent failure" not in path.read_text()


def test_attempt_file_cannot_be_overwritten(tmp_path):
    path = tmp_path / "attempt.jsonl"
    path.write_text("previous attempt")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path), "unused:invoke", "--stream-file", str(path)],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0
    assert path.read_text() == "previous attempt"


def test_call_handler_reuses_loop_and_imports_without_crossing_attempt_streams(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "resident_test_agent.py").write_text(
        "import asyncio\n"
        "calls = 0\n"
        "loop = None\n"
        "async def invoke(request, state_dir, *, emit):\n"
        "    global calls, loop\n"
        "    current = asyncio.get_running_loop()\n"
        "    if loop is None: loop = current\n"
        "    assert loop is current\n"
        "    calls += 1\n"
        "    emit(request['text'])\n"
        "    return {'calls': calls, 'state': str(state_dir)}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = {}
    exec(CALL_SCRIPT, namespace)
    # Restore sys.modules when the test ends while exercising real import caching.
    monkeypatch.setitem(sys.modules, "resident_test_agent", None)
    del sys.modules["resident_test_agent"]
    paths = [tmp_path / "streams" / f"attempt-{index}.jsonl" for index in range(2)]

    async def exercise():
        results = []
        for index, path in enumerate(paths):
            results.append(
                await namespace["call_handler"](
                    {"text": f"turn {index}"},
                    tmp_path / "state",
                    "resident_test_agent:invoke",
                    path,
                )
            )
        return results

    results = asyncio.run(exercise())
    assert [result["calls"] for result in results] == [1, 2]
    assert all(result["state"] == str(tmp_path / "state") for result in results)
    assert capsys.readouterr().out == ""
    for index, path in enumerate(paths):
        assert path.stat().st_mode & 0o777 == 0o600
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"type": "delta", "delta": f"turn {index}", "at": ANY},
            {"type": "end", "status": "completed"},
        ]


@pytest.mark.parametrize("value", [None, {"count": 1}, [1, "two"], "reply"])
def test_call_handler_supports_sync_without_emit_and_returns_original_result(
    tmp_path, monkeypatch, value
):
    module = ModuleType("plain_test_agent")
    module.invoke = lambda request, state_dir: value
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {}
    exec(CALL_SCRIPT, namespace)
    result = asyncio.run(namespace["call_handler"]({}, tmp_path, "plain_test_agent:invoke"))
    assert result is value


@pytest.mark.parametrize("failure", ["serialize", "cancel", "invalid_delta"])
def test_call_handler_records_failure_closes_stream_and_allows_next_call(
    tmp_path, monkeypatch, failure
):
    module = ModuleType("failing_test_agent")
    retained_emit = []

    async def invoke(request, state_dir, *, emit):
        retained_emit.append(emit)
        emit("partial")
        if failure == "serialize":
            return object()
        if failure == "cancel":
            raise asyncio.CancelledError()
        emit(42)

    module.invoke = invoke
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = {}
    exec(CALL_SCRIPT, namespace)
    path = tmp_path / "failed.jsonl"

    async def exercise():
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else TypeError):
            await namespace["call_handler"]({}, tmp_path, "failing_test_agent:invoke", path)
        # A retained callback must not write into another invocation's stream.
        with pytest.raises(ValueError, match="closed file"):
            retained_emit[0]("late output")
        module.invoke = lambda request, state_dir: {"reply": "recovered"}
        return await namespace["call_handler"](
            {}, tmp_path, "failing_test_agent:invoke", tmp_path / "next.jsonl"
        )

    assert asyncio.run(exercise()) == {"reply": "recovered"}
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"type": "delta", "delta": "partial", "at": ANY},
        {"type": "end", "status": "failed"},
    ]
    assert json.loads((tmp_path / "next.jsonl").read_text()) == {
        "type": "end",
        "status": "completed",
    }


def test_call_handler_records_import_failure(tmp_path):
    namespace = {}
    exec(CALL_SCRIPT, namespace)
    path = tmp_path / "missing.jsonl"
    with pytest.raises(ModuleNotFoundError):
        asyncio.run(
            namespace["call_handler"]({}, tmp_path, "nonexistent_runner_test_agent:invoke", path)
        )
    assert json.loads(path.read_text()) == {"type": "end", "status": "failed"}


def test_stream_decoder_handles_split_utf8():
    chunks = []
    decoder = StreamDecoder(chunks.append)
    data = (
        json.dumps({"type": "delta", "delta": "🌍"}, ensure_ascii=False)
        + "\n"
        + '{"type":"end","status":"completed"}\n'
    ).encode()
    for byte in data:
        decoder.feed(bytes([byte]))
    assert chunks == ["🌍"]
    assert decoder.ended.is_set()


@pytest.mark.parametrize(
    "data",
    [
        b"log line\n",
        b'{"type":"delta","delta":42}\n',
        b'{"type":"end","status":"unknown"}\n',
        b"x" * (1024 * 1024 + 1),
    ],
)
def test_stream_decoder_rejects_corruption(data):
    with pytest.raises(ValueError):
        StreamDecoder(lambda _: None).feed(data)
