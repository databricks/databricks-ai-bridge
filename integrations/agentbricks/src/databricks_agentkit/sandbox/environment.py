"""Versioned source publications and their runtime binding."""

ARTIFACT_FORMAT = "agentbricks.environment.v4"
PUBLISHED_FORMAT = "agentbricks.environment.v3"
SHARED_FORMAT = "agentbricks.environment.v2"
PREPARED_FORMAT = "agentbricks.prepared-environment.v1"


def on_demand(manifest):
    return manifest.get("format") in {ARTIFACT_FORMAT, PUBLISHED_FORMAT, SHARED_FORMAT}


def sandbox_scope(manifest):
    """Keep existing shared sandboxes addressable when upgrading the runtime."""
    return (
        "session"
        if manifest.get("format") in {ARTIFACT_FORMAT, PUBLISHED_FORMAT}
        else "environment"
    )


def bind_environment(orchestrator, manifest):
    """Configure scope without accessing a sandbox or claiming its ownership."""
    expected = "published" if on_demand(manifest) else "ready"
    if manifest.get("status") != expected:
        raise ValueError(f"Environment must be {expected} before serving")
    if manifest["namespace"] != orchestrator.namespace:
        raise ValueError("Environment and orchestrator namespaces differ")
    orchestrator.sandbox_scope = sandbox_scope(manifest)
    orchestrator.prepared_session_id = None
    orchestrator.environment_session_id = None
    if on_demand(manifest):
        orchestrator.environment_session_id = manifest["runtime_key"]
    else:
        orchestrator.prepared_session_id = manifest["preparation_id"]


def saved_bundle(control, manifest):
    """Use the immutable published bytes even if Workspace files have changed."""
    from databricks_agentkit.sandbox.bundle import Bundle

    agent = control.agent(manifest["agent_id"], with_bundle=True)
    if manifest.get("format") == ARTIFACT_FORMAT:
        artifact = agent.get("execution_artifact")
        if not artifact or artifact != manifest.get("execution_artifact"):
            raise ValueError("Environment artifact does not match its saved revision")
    if agent["source_hash"] != manifest["source_hash"]:
        raise ValueError("Environment source does not match its saved bundle")
    return Bundle(bytes(agent["bundle_data"]), agent["source_hash"], agent["file_count"])
