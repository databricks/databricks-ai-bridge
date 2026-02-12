"""
One-time setup script for Genie integration test resources.

Creates:
1. Schema: integration_testing.databricks_ai_bridge_genie_test
2. Table: orders (with deterministic sample data)
3. Genie Space connected to the test table

Usage:
    DATABRICKS_CONFIG_PROFILE=ai-oss-prod python scripts/setup_genie_test_resources.py

After running, set the Genie Space ID printed at the end as GENIE_SPACE_ID.
"""

import argparse
import json

from databricks.sdk import WorkspaceClient

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_genie_test"
TABLE = f"{CATALOG}.{SCHEMA}.orders"


def get_warehouse_id(workspace_client: WorkspaceClient) -> str:
    """Get the first available SQL warehouse ID."""
    warehouses = list(workspace_client.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found. Create one first.")
    wh = warehouses[0]
    print(f"  Using warehouse: {wh.name} ({wh.id})")
    return wh.id


def create_schema(workspace_client: WorkspaceClient):
    """Create the test schema (idempotent)."""
    print(f"Creating schema {CATALOG}.{SCHEMA}...")
    workspace_client.schemas.create(
        name=SCHEMA,
        catalog_name=CATALOG,
        comment="Test schema for Genie integration tests (databricks-ai-bridge)",
    )
    print(f"  Schema {CATALOG}.{SCHEMA} created.")


def create_table(workspace_client: WorkspaceClient, warehouse_id: str):
    """Create the orders test table with sample data (idempotent)."""
    print(f"Creating table {TABLE}...")

    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                order_id INT,
                customer_name STRING,
                order_date DATE,
                amount DECIMAL(10,2),
                region STRING,
                status STRING
            ) USING DELTA
        """,
        wait_timeout="30s",
    )

    # Only insert if table is empty (idempotent)
    count_result = workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=f"SELECT COUNT(*) AS cnt FROM {TABLE}",
        wait_timeout="30s",
    )
    data = count_result.result and count_result.result.data_array
    row_count = int(data[0][0]) if data else 0
    if row_count > 0:
        print(f"  Table {TABLE} already has {row_count} rows, skipping insert.")
        return

    workspace_client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=f"""
            INSERT INTO {TABLE} VALUES
            (1, 'Alice Johnson', '2024-01-15', 150.00, 'East', 'completed'),
            (2, 'Bob Smith', '2024-01-20', 250.50, 'West', 'completed'),
            (3, 'Carol White', '2024-02-01', 75.25, 'East', 'pending'),
            (4, 'David Brown', '2024-02-10', 320.00, 'West', 'completed'),
            (5, 'Eve Davis', '2024-02-15', 180.75, 'East', 'completed'),
            (6, 'Frank Miller', '2024-03-01', 95.00, 'West', 'pending'),
            (7, 'Grace Wilson', '2024-03-10', 410.25, 'East', 'completed'),
            (8, 'Henry Taylor', '2024-03-15', 200.00, 'West', 'completed'),
            (9, 'Ivy Anderson', '2024-04-01', 55.50, 'East', 'pending'),
            (10, 'Jack Thomas', '2024-04-10', 300.00, 'West', 'completed'),
            (11, 'Karen Lee', '2024-04-15', 125.75, 'East', 'completed'),
            (12, 'Leo Harris', '2024-05-01', 275.00, 'West', 'pending'),
            (13, 'Mia Clark', '2024-05-10', 190.50, 'East', 'completed'),
            (14, 'Noah Lewis', '2024-05-15', 350.25, 'West', 'completed'),
            (15, 'Olivia Walker', '2024-06-01', 110.00, 'East', 'pending'),
            (16, 'Peter Hall', '2024-06-10', 445.00, 'West', 'completed'),
            (17, 'Quinn Allen', '2024-06-15', 85.25, 'East', 'completed'),
            (18, 'Ruby Young', '2024-07-01', 230.50, 'West', 'pending'),
            (19, 'Sam King', '2024-07-10', 165.00, 'East', 'completed'),
            (20, 'Tina Wright', '2024-07-15', 390.75, 'West', 'completed')
        """,
        wait_timeout="30s",
    )
    print(f"  Table {TABLE} created with 20 rows.")


def create_genie_space(workspace_client: WorkspaceClient, warehouse_id: str) -> str:
    """Create a Genie Space connected to the test table."""
    print("Creating Genie Space...")

    serialized_space = json.dumps(
        {
            "version": 2,
            "config": {
                "sample_questions": [
                    {
                        "id": "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a1",
                        "question": ["What is the total amount by region?"],
                    },
                    {
                        "id": "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a2",
                        "question": ["How many orders are there?"],
                    },
                    {
                        "id": "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a3",
                        "question": ["What is the average amount by status?"],
                    },
                ],
            },
            "data_sources": {
                "tables": [
                    {
                        "identifier": TABLE,
                        "description": [
                            "Sample orders table with customer name, order date, amount, region, and status."
                        ],
                    },
                ],
            },
            "instructions": {
                "text_instructions": [
                    {
                        "id": "b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b1",
                        "content": [
                            "This table contains sample order data for integration testing. "
                            "Use SUM for total amounts and AVG for averages."
                        ],
                    },
                ],
            },
        }
    )

    space = workspace_client.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=serialized_space,
        title="AI Bridge Integration Test",
        description=(
            "Genie space for databricks-ai-bridge integration testing. "
            "Connected to a sample orders table with customer, region, status, and amount data."
        ),
    )
    print(f"  Genie Space created: {space.space_id}")
    return space.space_id


def main():
    parser = argparse.ArgumentParser(description="Setup Genie integration test resources")
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile to use (e.g. ai-oss-prod)",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.profile:
        kwargs["profile"] = args.profile

    workspace_client = WorkspaceClient(**kwargs)
    print(f"Connected to: {workspace_client.config.host}")

    warehouse_id = get_warehouse_id(workspace_client)

    try:
        create_schema(workspace_client)
    except Exception as e:
        if "SCHEMA_ALREADY_EXISTS" in str(e):
            print(f"  Schema {CATALOG}.{SCHEMA} already exists, skipping.")
        else:
            raise

    create_table(workspace_client, warehouse_id)
    space_id = create_genie_space(workspace_client, warehouse_id)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"Genie Space ID: {space_id}")
    print(f"Table: {TABLE}")
    print(f"\nTo run tests:")
    print(f"  export GENIE_SPACE_ID={space_id}")
    print(f"  export DATABRICKS_CONFIG_PROFILE=ai-oss-prod")
    print(f"  RUN_GENIE_INTEGRATION_TESTS=1 uv run --group tests pytest tests/integration_tests/genie/ -v")


if __name__ == "__main__":
    main()
