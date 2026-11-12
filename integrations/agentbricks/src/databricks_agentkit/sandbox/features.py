"""Non-secret capability declarations from the published agent source."""

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib


def source_features(manifest: bytes | None) -> dict:
    document = tomllib.loads(manifest.decode()) if manifest else {}
    store = document.get("session_store", {}).get("name")
    experiment = document.get("tracing", {}).get("experiment_name")
    framework = document.get("agent", {}).get("framework", "custom")
    return {
        "framework": framework,
        "sessions": {"mode": "managed", "store": store}
        if store
        else {"mode": "sandbox-sqlite" if framework in {"langgraph", "openai"} else "custom"},
        "tracing": {"configured": bool(experiment), "experiment_name": experiment},
        "tools": [item["id"] for item in document.get("tools", []) if item.get("id")],
    }
