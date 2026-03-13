#!/usr/bin/env python3
"""Simplified start script for CI deployment (backend only, no UI)."""

import argparse
import subprocess
import sys
import threading

from dotenv import load_dotenv


def main():
    load_dotenv(dotenv_path=".env", override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ui", action="store_true", default=True)
    args, backend_args = parser.parse_known_args()

    cmd = ["uv", "run", "start-server"] + backend_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def monitor():
        for line in iter(proc.stdout.readline, ""):
            print(line.rstrip())  # noqa: T201

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
