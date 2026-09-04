"""Long-running server and checkpoint lifecycle shared by both Apps."""

import os
from contextlib import asynccontextmanager

from databricks_langchain import AsyncCheckpointSaver
from fastapi import FastAPI

from databricks_ai_bridge.long_running import LongRunningAgentServer, ResumeStrategy

_checkpointer: AsyncCheckpointSaver | None = None


def get_checkpointer() -> AsyncCheckpointSaver:
    if _checkpointer is None:
        raise RuntimeError("LangGraph checkpointer is not initialized")
    return _checkpointer


def create_app(resume_strategy: ResumeStrategy) -> FastAPI:
    agent_server = LongRunningAgentServer(
        "ResponsesAgent",
        db_autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
        resume_strategy=resume_strategy,
    )
    app = agent_server.app
    runtime_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        global _checkpointer
        async with AsyncCheckpointSaver(
            autoscaling_endpoint=os.environ["LAKEBASE_AUTOSCALING_ENDPOINT"],
            schema=os.getenv(
                "LAKEBASE_CHECKPOINT_SCHEMA",
                "langgraph_agent_checkpoints",
            ),
        ) as checkpointer:
            await checkpointer.setup()
            _checkpointer = checkpointer
            try:
                async with runtime_lifespan(application):
                    yield
            finally:
                _checkpointer = None

    app.router.lifespan_context = lifespan
    return app
