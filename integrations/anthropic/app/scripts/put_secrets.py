"""Store the App's secrets in its secret scope without printing them.

    uv run python app/scripts/put_secrets.py --profile <profile> --scope <scope> [--key-env VAR] [--rotate-hmac]

Writes ``env-s-key`` (the service-mode environment key, read from the environment variable ``--key-env``,
default ``ANTHROPIC_ENVIRONMENT_KEY``, or prompted for without echo if it is unset) and, if missing or with
--rotate-hmac, ``sandbox-id-hmac-key`` (32 random bytes, hex). Rotating the HMAC key orphans existing
sandbox names, so only rotate when no sessions are running. Values go to the API only, never to stdout,
the command line or disk.
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

import databricks_anthropic  # noqa: F401  # sets the SDK's User-Agent product before the client is built


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--profile", required=True)
    p.add_argument("--scope", required=True)
    p.add_argument(
        "--key-env",
        default="ANTHROPIC_ENVIRONMENT_KEY",
        help="variable holding the environment key",
    )
    p.add_argument("--rotate-hmac", action="store_true")
    args = p.parse_args()

    key = os.environ.get(args.key_env) or getpass.getpass(f"{args.key_env} (not echoed): ")
    if not key:
        raise SystemExit(f"no environment key: set {args.key_env} or enter it at the prompt")
    w = WorkspaceClient(config=Config(profile=args.profile))  # global product (set on import)
    w.secrets.put_secret(args.scope, "env-s-key", string_value=key)
    print(f"stored env-s-key in {args.scope}")
    existing = {s.key for s in w.secrets.list_secrets(args.scope)}
    if args.rotate_hmac or "sandbox-id-hmac-key" not in existing:
        w.secrets.put_secret(args.scope, "sandbox-id-hmac-key", string_value=secrets.token_hex(32))
        print(f"stored a new sandbox-id-hmac-key in {args.scope}")
    else:
        print("kept the existing sandbox-id-hmac-key")


if __name__ == "__main__":
    main()
