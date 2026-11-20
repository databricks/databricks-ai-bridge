"""Input and tool authorization shared by first-party sandbox adapters."""

import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib


@contextmanager
def measure(timings, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - started) * 1000, 2)


@lru_cache(maxsize=1)
def configure_project(project):
    """Template configuration is process startup work, independent of each turn."""
    project.configure()
    # MLflow's first span end can synchronously fetch pricing from GitHub.
    # Use its bundled catalog by default; preserve project/env configuration.
    os.environ.setdefault("MLFLOW_MODEL_CATALOG_URI", "")


@lru_cache(maxsize=1)
def app_workspace_client():
    """One refreshable app-auth client per session worker, never request-user auth."""
    from databricks_agentkit import workspace_client

    return workspace_client()


def prepare(request, state_dir, default_model):
    value = request["input"]
    message = value.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("input.message must be a non-empty string")
    model = value.get("model") or default_model
    if not isinstance(model, str):
        raise ValueError("input.model must be a model service name")
    prepare_source(state_dir)
    return message, model


def prepare_source(state_dir):
    # Agent code is installed in <source>/.venv, with cwd outside the source.
    # Tool bindings still need the source manifest, never the App's manifest.
    os.environ["AGENTBRICKS_PROJECT_ROOT"] = str(Path(sys.prefix).parent)
    manifest = Path(os.environ["AGENTBRICKS_PROJECT_ROOT"]) / "agent.toml"
    if manifest.is_file():
        configure_bindings(tomllib.loads(manifest.read_text()))
    state_dir.mkdir(parents=True, exist_ok=True)


def configure_bindings(document):
    """Resolve source bindings in the agent process, without forwarding App credentials."""
    if name := document.get("session_store", {}).get("name"):
        os.environ.setdefault("AGENT_SESSION_STORE", name)
    if name := document.get("tracing", {}).get("experiment_name"):
        import mlflow

        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        if os.getenv("MLFLOW_EXPERIMENT_NAME") != name or not os.getenv("MLFLOW_EXPERIMENT_ID"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            experiment = mlflow.set_experiment(experiment_name=name)
            os.environ["MLFLOW_EXPERIMENT_ID"] = experiment.experiment_id
            os.environ["MLFLOW_EXPERIMENT_NAME"] = name


def trace_result(span):
    """Return trace identity immediately; the resident exporter runs asynchronously."""
    if span is None:
        return None

    trace_id = span.trace_id
    result = {"trace_id": trace_id, "url": None}
    try:
        host = app_workspace_client().config.host.rstrip("/")
        experiment = quote(os.getenv("MLFLOW_EXPERIMENT_ID", ""), safe="")
        if experiment:
            suffix = quote(trace_id.rsplit("/", 1)[-1], safe="")
            result["url"] = f"{host}/ml/experiments/{experiment}/traces/{suffix}"
    except Exception as exc:
        # Agent execution may already have performed tool side effects. Keep its
        # outcome and saved trace ID even if exporting or constructing a link fails.
        logging.getLogger(__name__).warning("Trace finalization failed: %s", type(exc).__name__)
        result["export_status"] = "unverified"
    return result


def shutdown_session(state_dir):
    """Drain trace exports once at worker shutdown, outside individual turns."""
    import mlflow

    mlflow.flush_trace_async_logging()


def workspace_client_for(mode):
    if mode != "app":
        raise RuntimeError(
            "The sandbox adapter supports app-auth tools only. Request-user "
            "authorization is not forwarded; configure this demo with --auth app."
        )
    return app_workspace_client()


def text_content(content):
    """Keep text blocks; exclude reasoning/tool-call payloads from chat previews."""
    if isinstance(content, str):
        # databricks-langchain 0.21.0 serializes signed Claude reasoning as JSON
        # strings. Reuse the history codec without modifying checkpoint content.
        from databricks_agentkit.langgraph.messages import delta_contents

        return "".join(part for part in delta_contents(content) if isinstance(part, str))
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""
