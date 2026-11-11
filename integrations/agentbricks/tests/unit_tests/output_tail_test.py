"""Retention and relay regressions using synthetic output and local pipes."""

import os
import subprocess
import sys
import threading
import time

import pytest

from databricks_agentbricks.clients.output_tail import OutputTail, StreamRelay


@pytest.mark.parametrize("chunk_size", [1, 7, 8192])
def test_pem_output_is_omitted_even_when_the_header_leaves_the_tail(chunk_size):
    output = (
        b"-----BEGIN PRIVATE KEY-----\n"
        + b"synthetic-key-body\n" * 400
        + b"-----END PRIVATE KEY-----\nstartup failed\n"
    )
    tail = OutputTail()
    for offset in range(0, len(output), chunk_size):
        tail.append(output[offset : offset + chunk_size])
    assert tail.text() == "[REDACTED]"


@pytest.mark.parametrize("chunk_size", [1, 7, 8192])
@pytest.mark.parametrize(
    "opening, closing",
    [(b"credentials = {\n", b"}\n"), (b"tokens = [\n", b"]\n"), (b'password="', b'"\n')],
)
def test_truncated_structured_values_do_not_outlive_their_credential_context(
    chunk_size, opening, closing
):
    output = opening + b" \n" * 2000 + b"synthetic-opaque-value\n" + closing
    tail = OutputTail()
    for offset in range(0, len(output), chunk_size):
        tail.append(output[offset : offset + chunk_size])
    assert tail.text() == "[REDACTED]"
    tail.append(b"later output\n" * 400)
    assert tail.text() == "[REDACTED]"


def test_short_structured_output_keeps_its_context_for_downstream_scrubbing():
    output = b'credentials = {"value": "synthetic-opaque-value"}\n'
    tail = OutputTail()
    tail.append(output)
    assert tail.text() == output.decode()


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux pipe sizing")
def test_slow_open_destination_preserves_every_byte_after_child_exit():
    import fcntl

    payload = b"x" * 10000 + b"\nFINAL-FAILURE\n"
    read_fd, write_fd = os.pipe()
    fcntl.fcntl(write_fd, fcntl.F_SETPIPE_SZ, 4096)
    received = bytearray()
    child_exited = threading.Event()

    def consume():
        child_exited.wait(timeout=5)
        time.sleep(0.5)
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            while data := stream.read(4096):
                received.extend(data)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    try:
        with subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 10000 + b'\\nFINAL-FAILURE\\n')",
            ],
            stdout=subprocess.PIPE,
            bufsize=0,
            pipesize=4096,
        ) as proc:
            assert proc.stdout is not None
            relay = StreamRelay(proc.stdout, write_fd)
            try:
                assert proc.wait(timeout=5) == 0
                child_exited.set()
                relay.finish()
                assert not relay._thread.is_alive()
                assert relay.tail.text() == "FINAL-FAILURE\n"
            finally:
                child_exited.set()
                proc.kill()
    finally:
        os.close(write_fd)
        consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert bytes(received) == payload
