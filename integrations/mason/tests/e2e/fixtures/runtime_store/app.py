"""Live fixture: real Mason durability, deterministic echo, no model or other stores."""

import os

import uvicorn
from sqlalchemy import text

from databricks_mason import AgentApp, InvocationContext
from databricks_mason.runtime.durability.lakebase_runtime_store import LakebaseDurableRuntimeStore
from databricks_mason.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
    runtime_store_from_environment,
)
from databricks_mason.runtime.types import JsonValue

configured_store = runtime_store_from_environment()
if not isinstance(configured_store, LakebaseDurableRuntimeStore):
    raise RuntimeError(
        "Deploy this fixture with Mason: the test requires a Lakebase Runtime Store."
    )
store: LakebaseDurableRuntimeStore = configured_store
app = AgentApp(runtime_store=store)


@app.invoke
@app.recover
async def invoke(value: JsonValue, context: InvocationContext) -> JsonValue:
    await context.emit({"marker": "runtime-store-e2e", "input": value})
    return {"echo": value, "invocation_id": context.invocation_id}


@app.get("/api/runtime-store-proof")
async def proof() -> dict:
    """Return database/schema/table ownership as the app SP; never return credentials."""
    async with store._engine.connect() as connection:
        identity = (
            (
                await connection.execute(
                    text(
                        "SELECT current_user AS role, current_database() AS database, "
                        "pg_get_userbyid(datdba) AS database_owner, "
                        "has_database_privilege(current_user, current_database(), 'CREATE') AS can_create "
                        "FROM pg_database WHERE datname = current_database()"
                    )
                )
            )
            .mappings()
            .one()
        )
        schema_owner = (
            await connection.execute(
                text("SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname = :schema"),
                {"schema": os.environ[RUNTIME_STORE_SCHEMA_ENV]},
            )
        ).scalar_one()
        tables = (
            (
                await connection.execute(
                    text(
                        "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = :schema "
                        "AND tablename IN ('invocations', 'invocation_events') ORDER BY tablename"
                    ),
                    {"schema": os.environ[RUNTIME_STORE_SCHEMA_ENV]},
                )
            )
            .mappings()
            .all()
        )
    assert identity["role"] == identity["database_owner"] == os.environ[RUNTIME_STORE_USERNAME_ENV]
    assert identity["database"] == os.environ[RUNTIME_STORE_DATABASE_ENV]
    assert identity["can_create"]
    assert schema_owner == identity["role"]
    assert [table["tablename"] for table in tables] == ["invocation_events", "invocations"]
    assert all(table["tableowner"] == identity["role"] for table in tables)
    return {**dict(identity), "schema_owner": schema_owner, "tables": [dict(row) for row in tables]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["DATABRICKS_APP_PORT"]))
