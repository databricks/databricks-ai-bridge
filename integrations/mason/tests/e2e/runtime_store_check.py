"""Check a deployed Runtime Store fixture; the caller deploys and deletes it with Mason."""

import argparse
import json
import pathlib
import time
import uuid

import requests

from databricks_mason import lakebase_runtime_store
from databricks_mason._api_client import _MasonApiClient


def wait_for_app(http: requests.Session, url: str) -> dict:
    # Apps reports compute ACTIVE before the application finishes starting after a restart.
    deadline = time.monotonic() + 180
    while True:
        try:
            response = http.get(f"{url}/api/runtime-store-proof", timeout=15)
        except (requests.ConnectionError, requests.Timeout):
            if time.monotonic() >= deadline:
                raise
        else:
            if response.status_code not in (502, 503) or time.monotonic() >= deadline:
                response.raise_for_status()
                return response.json()
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--app", required=True, help="Full agent-mason-prefixed deployment name")
    parser.add_argument("--invocation-id", type=uuid.UUID, required=True)
    parser.add_argument(
        "--read-only", action="store_true", help="Verify persisted data after restart"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    api = _MasonApiClient(profile=args.profile)
    app = api._w.apps.get(args.app)
    assert app.url and app.service_principal_client_id
    store_id = lakebase_runtime_store.runtime_store_id(args.app, app.service_principal_client_id)
    resource = api.get_runtime_store(store_id)
    backend = lakebase_runtime_store.backend_from_api(
        args.app, store_id, app.service_principal_client_id, resource
    )
    assert backend.project == "databricks-internal-custom-agents"
    assert backend.database != store_id
    invocation_id = str(args.invocation_id)
    payload = {"marker": "runtime-store-e2e", "id": invocation_id}
    expected_output = {"echo": payload, "invocation_id": invocation_id}

    with requests.Session() as http:
        # Credentials stay in memory. The app itself uses its injected SP credentials for Postgres.
        http.headers.update(api._w.config.authenticate())
        url = app.url.rstrip("/")
        proof = wait_for_app(http, url)
        assert proof["role"] == proof["database_owner"] == app.service_principal_client_id
        assert proof["database"] == backend.database
        assert proof["schema_owner"] == app.service_principal_client_id
        assert proof["can_create"]
        assert {table["tablename"] for table in proof["tables"]} == {
            "invocations",
            "invocation_events",
        }
        assert all(
            table["tableowner"] == app.service_principal_client_id for table in proof["tables"]
        )
        if not args.read_only:
            created = http.post(
                f"{url}/api/invocations", json={"id": invocation_id, "input": payload}, timeout=60
            )
            created.raise_for_status()
            assert created.json()["output"] == expected_output
        read = http.get(f"{url}/api/invocations/{invocation_id}", timeout=60)
        read.raise_for_status()
        assert read.json()["status"] == "completed"
        assert read.json()["output"] == expected_output
        events = http.get(f"{url}/api/invocations/{invocation_id}/events", timeout=60)
        events.raise_for_status()
        application_events = [
            json.loads(line.removeprefix("data: "))
            for line in events.text.splitlines()
            if line.startswith("data: ")
        ]
        assert {"marker": "runtime-store-e2e", "input": payload} in application_events

    evidence = {
        "profile": args.profile,
        "app": args.app,
        "runtime_store": resource,
        "ownership": proof,
        "invocation": read.json(),
        "events": application_events,
        "read_only": args.read_only,
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
