"""
Integration tests for LakebaseClient permission granting.

These tests require a live Lakebase instance and appropriate credentials.
They are NOT run in CI by default - set LAKEBASE_INTEGRATION_TESTS=1 to enable.

Environment Variables:
======================
Required for all tests:
    LAKEBASE_INSTANCE_NAME    - Name of your Lakebase instance
    TEST_SERVICE_PRINCIPAL    - A REAL service principal UUID from your workspace
                                (find in Admin Console > Identity and access > Service principals)
    LAKEBASE_INTEGRATION_TESTS - Set to "1" to enable integration tests

Optional for authenticating as different users:
    DATABRICKS_HOST           - Workspace URL (e.g., https://your-workspace.databricks.com)
    DATABRICKS_TOKEN          - OAuth token for the main test user (should have 'CAN MANAGE')

Optional for testing permission errors (tests will be SKIPPED if not set):
    NO_ROLE_USER_TOKEN        - OAuth token for a user who has NO PostgreSQL role in Lakebase.
                                This user can authenticate to Databricks but create_role() was
                                never called for them. Tests verify "role does not exist" errors.

    LIMITED_PERMISSION_USER_TOKEN - OAuth token for a user who HAS a PostgreSQL role but lacks
                                    GRANT permissions. This user can connect but cannot grant
                                    privileges to others. Tests verify PermissionError handling.

Basic test run:
    export LAKEBASE_INSTANCE_NAME=your-lakebase-instance
    export TEST_SERVICE_PRINCIPAL=your-test-sp-uuid
    export LAKEBASE_INTEGRATION_TESTS=1
    pytest tests/databricks_ai_bridge/test_lakebase_integration.py -v

Test with specific OAuth token:
    export DATABRICKS_HOST=https://your-workspace.databricks.com
    export DATABRICKS_TOKEN=your-oauth-token
    pytest tests/databricks_ai_bridge/test_lakebase_integration.py -v

Test "no role" error scenario:
    export NO_ROLE_USER_TOKEN=oauth-token-for-user-without-database-role
    pytest tests/databricks_ai_bridge/test_lakebase_integration.py::TestNoRoleUserErrors -v

Test "limited permission" error scenario:
    export LIMITED_PERMISSION_USER_TOKEN=oauth-token-for-user-with-role-but-no-grant
    pytest tests/databricks_ai_bridge/test_lakebase_integration.py::TestLimitedPermissionUserErrors -v

Example to run all tests:
    export DATABRICKS_HOST=[host]
    export DATABRICKS_TOKEN=[super-user-manage-oauth-token]
    export LIMITED_PERMISSION_USER_TOKEN=[role-no-permission-oauth-token]
    export NO_ROLE_USER_TOKEN=[no-role-oauth-token]
    export LAKEBASE_INSTANCE_NAME=[lakebase]
    export TEST_SERVICE_PRINCIPAL=[sp-uuid]
    export LAKEBASE_INTEGRATION_TESTS=1
    pytest tests/databricks_ai_bridge/test_lakebase_integration.py -v
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this module if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("LAKEBASE_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set LAKEBASE_INTEGRATION_TESTS=1 to enable.",
)

pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

from databricks.sdk import WorkspaceClient

from databricks_ai_bridge.lakebase import (
    LakebaseClient,
    LakebasePool,
    SchemaPrivilege,
    SequencePrivilege,
    TablePrivilege,
)


def create_workspace_client_with_token(token: str | None = None) -> WorkspaceClient:
    """
    Create a WorkspaceClient, optionally with a specific OAuth token.

    If token is provided, creates a client using that token.
    Otherwise, uses default credential chain (env vars, .databrickscfg, etc.)

    :param token: Optional OAuth token to authenticate with.
    :return: Configured WorkspaceClient.
    """
    if token:
        host = os.environ.get("DATABRICKS_HOST")
        if not host:
            raise ValueError(
                "DATABRICKS_HOST must be set when using a custom token. "
                "Example: export DATABRICKS_HOST=https://your-workspace.databricks.com"
            )
        return WorkspaceClient(host=host, token=token)
    return WorkspaceClient()


@pytest.fixture(scope="module")
def instance_name():
    """Get Lakebase instance name from environment."""
    name = os.environ.get("LAKEBASE_INSTANCE_NAME")
    if not name:
        pytest.skip("LAKEBASE_INSTANCE_NAME not set")
    return name


@pytest.fixture(scope="module")
def test_service_principal():
    """Get test service principal UUID from environment."""
    sp = os.environ.get("TEST_SERVICE_PRINCIPAL")
    if not sp:
        pytest.skip("TEST_SERVICE_PRINCIPAL not set")
    return sp


@pytest.fixture(scope="module")
def test_grantee(client, test_service_principal):
    """
    Ensure the test service principal role exists before running grant tests.

    This fixture creates the role if it doesn't exist, making it safe to run
    grant tests without manual setup.
    """
    # Create the role - this is idempotent (logs info if already exists)
    client.create_role(test_service_principal, "SERVICE_PRINCIPAL")
    return test_service_principal


@pytest.fixture(scope="module")
def workspace_client():
    """
    Create a WorkspaceClient for testing.

    Uses DATABRICKS_TOKEN if set, otherwise uses default credential chain.
    This allows testing as different users by passing different OAuth tokens.

    Usage:
        # Test with default credentials
        pytest tests/...

        # Test as a specific user
        export DATABRICKS_HOST=https://your-workspace.databricks.com
        export DATABRICKS_TOKEN=dapi_your_oauth_token
        pytest tests/...
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    return create_workspace_client_with_token(token)


@pytest.fixture(scope="module")
def client(instance_name, workspace_client):
    """
    Create a LakebaseClient for testing.

    Uses the workspace_client fixture which respects DATABRICKS_TOKEN env var,
    allowing tests to run as different users.
    """
    pool = LakebasePool(
        instance_name=instance_name,
        workspace_client=workspace_client,
    )
    client = LakebaseClient(pool=pool)
    yield client
    client.close()
    pool.close()


# =============================================================================
# Validation Error Tests (don't require live Lakebase)
# =============================================================================


class TestValidationErrors:
    """Test validation errors are raised with helpful messages."""

    def test_grant_schema_empty_schemas_raises_error(self, client, test_service_principal):
        """grant_schema should raise ValueError when schemas is empty."""
        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_schema(
                grantee=test_service_principal,
                privileges=[SchemaPrivilege.USAGE],
                schemas=[],
            )

    def test_grant_schema_empty_privileges_raises_error(
        self, client, test_service_principal
    ):
        """grant_schema should raise ValueError when privileges is empty."""
        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_schema(
                grantee=test_service_principal,
                privileges=[],
                schemas=["public"],
            )

    def test_grant_all_tables_empty_schemas_raises_error(
        self, client, test_service_principal
    ):
        """grant_all_tables_in_schema should raise ValueError when schemas is empty."""
        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_all_tables_in_schema(
                grantee=test_service_principal,
                privileges=[TablePrivilege.SELECT],
                schemas=[],
            )

    def test_grant_all_tables_empty_privileges_raises_error(
        self, client, test_service_principal
    ):
        """grant_all_tables_in_schema should raise ValueError when privileges is empty."""
        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_all_tables_in_schema(
                grantee=test_service_principal,
                privileges=[],
                schemas=["public"],
            )

    def test_grant_table_empty_tables_raises_error(self, client, test_service_principal):
        """grant_table should raise ValueError when tables is empty."""
        with pytest.raises(ValueError, match="'tables' cannot be empty"):
            client.grant_table(
                grantee=test_service_principal,
                privileges=[TablePrivilege.SELECT],
                tables=[],
            )

    def test_grant_table_empty_privileges_raises_error(
        self, client, test_service_principal
    ):
        """grant_table should raise ValueError when privileges is empty."""
        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_table(
                grantee=test_service_principal,
                privileges=[],
                tables=["public.users"],
            )

    def test_grant_all_sequences_empty_schemas_raises_error(
        self, client, test_service_principal
    ):
        """grant_all_sequences_in_schema should raise ValueError when schemas is empty."""
        with pytest.raises(ValueError, match="'schemas' cannot be empty"):
            client.grant_all_sequences_in_schema(
                grantee=test_service_principal,
                privileges=[SequencePrivilege.USAGE],
                schemas=[],
            )

    def test_grant_all_sequences_empty_privileges_raises_error(
        self, client, test_service_principal
    ):
        """grant_all_sequences_in_schema should raise ValueError when privileges is empty."""
        with pytest.raises(ValueError, match="'privileges' cannot be empty"):
            client.grant_all_sequences_in_schema(
                grantee=test_service_principal,
                privileges=[],
                schemas=["public"],
            )

    def test_grant_table_invalid_table_format_raises_error(
        self, client, test_service_principal
    ):
        """grant_table should raise ValueError for invalid table name format."""
        with pytest.raises(ValueError, match="Invalid table format"):
            client.grant_table(
                grantee=test_service_principal,
                privileges=[TablePrivilege.SELECT],
                tables=["schema."],  # Invalid: missing table name
            )

    def test_grant_table_empty_table_name_raises_error(
        self, client, test_service_principal
    ):
        """grant_table should raise ValueError for empty table name."""
        with pytest.raises(ValueError, match="Table name cannot be empty"):
            client.grant_table(
                grantee=test_service_principal,
                privileges=[TablePrivilege.SELECT],
                tables=[""],
            )

    def test_create_role_empty_identity_raises_error(self, client):
        """create_role should raise ValueError for empty identity name."""
        with pytest.raises(ValueError, match="identity_name cannot be empty"):
            client.create_role("", "SERVICE_PRINCIPAL")

    def test_create_role_whitespace_identity_raises_error(self, client):
        """create_role should raise ValueError for whitespace-only identity name."""
        with pytest.raises(ValueError, match="identity_name cannot be empty"):
            client.create_role("   ", "SERVICE_PRINCIPAL")


# =============================================================================
# Schema Privilege Matrix Tests
# =============================================================================


class TestSchemaPrivilegeMatrix:
    """Test all SchemaPrivilege enum values."""

    @pytest.mark.parametrize(
        "privilege",
        [
            SchemaPrivilege.USAGE,
            SchemaPrivilege.CREATE,
            SchemaPrivilege.ALL,
        ],
    )
    def test_grant_schema_single_privilege(self, client, test_grantee, privilege):
        """Test granting each individual schema privilege."""
        # This will succeed if the schema exists and user has permission
        client.grant_schema(
            grantee=test_grantee,
            privileges=[privilege],
            schemas=["public"],
        )

    def test_grant_schema_multiple_privileges(self, client, test_grantee):
        """Test granting multiple schema privileges at once."""
        client.grant_schema(
            grantee=test_grantee,
            privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
            schemas=["public"],
        )

    def test_grant_schema_all_privileges(self, client, test_grantee):
        """Test granting ALL PRIVILEGES on schema."""
        client.grant_schema(
            grantee=test_grantee,
            privileges=[SchemaPrivilege.ALL],
            schemas=["public"],
        )


# =============================================================================
# Table Privilege Matrix Tests
# =============================================================================


class TestTablePrivilegeMatrix:
    """Test all TablePrivilege enum values."""

    @pytest.mark.parametrize(
        "privilege",
        [
            TablePrivilege.SELECT,
            TablePrivilege.INSERT,
            TablePrivilege.UPDATE,
            TablePrivilege.DELETE,
            TablePrivilege.TRUNCATE,
            TablePrivilege.REFERENCES,
            TablePrivilege.TRIGGER,
            TablePrivilege.ALL,
        ],
    )
    def test_grant_all_tables_single_privilege(self, client, test_grantee, privilege):
        """Test granting each individual table privilege on all tables."""
        client.grant_all_tables_in_schema(
            grantee=test_grantee,
            privileges=[privilege],
            schemas=["public"],
        )

    def test_grant_all_tables_common_privileges(self, client, test_grantee):
        """Test granting common table privileges (SELECT, INSERT, UPDATE, DELETE)."""
        client.grant_all_tables_in_schema(
            grantee=test_grantee,
            privileges=[
                TablePrivilege.SELECT,
                TablePrivilege.INSERT,
                TablePrivilege.UPDATE,
                TablePrivilege.DELETE,
            ],
            schemas=["public"],
        )

    def test_grant_all_tables_all_privileges(self, client, test_grantee):
        """Test granting ALL PRIVILEGES on all tables."""
        client.grant_all_tables_in_schema(
            grantee=test_grantee,
            privileges=[TablePrivilege.ALL],
            schemas=["public"],
        )


# =============================================================================
# Specific Table Grant Tests
# =============================================================================


class TestTableGrants:
    """Test granting privileges on specific tables."""

    @pytest.fixture
    def test_table(self, client):
        """Create a test table for grant tests."""
        table_name = "test_grant_table"
        client.execute(
            f"CREATE TABLE IF NOT EXISTS public.{table_name} (id SERIAL PRIMARY KEY, data TEXT)"
        )
        yield f"public.{table_name}"
        # Cleanup
        client.execute(f"DROP TABLE IF EXISTS public.{table_name}")

    @pytest.mark.parametrize(
        "privilege",
        [
            TablePrivilege.SELECT,
            TablePrivilege.INSERT,
            TablePrivilege.UPDATE,
            TablePrivilege.DELETE,
            TablePrivilege.ALL,
        ],
    )
    def test_grant_table_single_privilege(
        self, client, test_grantee, test_table, privilege
    ):
        """Test granting each privilege on a specific table."""
        client.grant_table(
            grantee=test_grantee,
            privileges=[privilege],
            tables=[test_table],
        )

    def test_grant_table_multiple_tables(self, client, test_grantee, test_table):
        """Test granting privileges on multiple tables at once."""
        # Create another test table
        client.execute(
            "CREATE TABLE IF NOT EXISTS public.test_grant_table2 (id SERIAL PRIMARY KEY)"
        )
        try:
            client.grant_table(
                grantee=test_grantee,
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT],
                tables=[test_table, "public.test_grant_table2"],
            )
        finally:
            client.execute("DROP TABLE IF EXISTS public.test_grant_table2")

    def test_grant_table_without_schema_prefix(self, client, test_grantee):
        """Test granting privileges on table without schema prefix."""
        # This assumes the table exists in the search_path
        client.execute(
            "CREATE TABLE IF NOT EXISTS test_no_schema (id SERIAL PRIMARY KEY)"
        )
        try:
            client.grant_table(
                grantee=test_grantee,
                privileges=[TablePrivilege.SELECT],
                tables=["test_no_schema"],
            )
        finally:
            client.execute("DROP TABLE IF EXISTS test_no_schema")


# =============================================================================
# Sequence Privilege Matrix Tests
# =============================================================================


class TestSequencePrivilegeMatrix:
    """Test all SequencePrivilege enum values."""

    @pytest.mark.parametrize(
        "privilege",
        [
            SequencePrivilege.USAGE,
            SequencePrivilege.SELECT,
            SequencePrivilege.UPDATE,
            SequencePrivilege.ALL,
        ],
    )
    def test_grant_all_sequences_single_privilege(
        self, client, test_grantee, privilege
    ):
        """Test granting each individual sequence privilege."""
        client.grant_all_sequences_in_schema(
            grantee=test_grantee,
            privileges=[privilege],
            schemas=["public"],
        )

    def test_grant_all_sequences_multiple_privileges(self, client, test_grantee):
        """Test granting multiple sequence privileges at once."""
        client.grant_all_sequences_in_schema(
            grantee=test_grantee,
            privileges=[
                SequencePrivilege.USAGE,
                SequencePrivilege.SELECT,
                SequencePrivilege.UPDATE,
            ],
            schemas=["public"],
        )

    def test_grant_all_sequences_all_privileges(self, client, test_grantee):
        """Test granting ALL PRIVILEGES on all sequences."""
        client.grant_all_sequences_in_schema(
            grantee=test_grantee,
            privileges=[SequencePrivilege.ALL],
            schemas=["public"],
        )


# =============================================================================
# Role Creation Tests
# =============================================================================


class TestCreateRole:
    """Test create_role functionality."""

    def test_create_role_service_principal(self, client, test_service_principal):
        """Test creating a role for a service principal."""
        # This may return None if the role already exists
        result = client.create_role(test_service_principal, "SERVICE_PRINCIPAL")
        # Result is either the creation result or None (if already exists)
        assert result is None or isinstance(result, list)

    def test_create_role_idempotent(self, client, test_service_principal):
        """Test that create_role is idempotent (can be called multiple times)."""
        # First call
        client.create_role(test_service_principal, "SERVICE_PRINCIPAL")
        # Second call should not raise, just log that role exists
        result = client.create_role(test_service_principal, "SERVICE_PRINCIPAL")
        assert result is None  # Returns None when role already exists


# =============================================================================
# Multiple Schema Tests
# =============================================================================


class TestMultipleSchemas:
    """Test granting privileges across multiple schemas."""

    @pytest.fixture
    def test_schemas(self, client):
        """Create test schemas for multi-schema tests."""
        schemas = ["test_schema_1", "test_schema_2"]
        for schema in schemas:
            client.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        yield schemas
        # Cleanup
        for schema in schemas:
            client.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")

    def test_grant_schema_multiple_schemas(
        self, client, test_grantee, test_schemas
    ):
        """Test granting schema privileges on multiple schemas."""
        client.grant_schema(
            grantee=test_grantee,
            privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
            schemas=test_schemas,
        )

    def test_grant_all_tables_multiple_schemas(
        self, client, test_grantee, test_schemas
    ):
        """Test granting table privileges across multiple schemas."""
        # Create tables in test schemas
        for schema in test_schemas:
            client.execute(
                f"CREATE TABLE IF NOT EXISTS {schema}.test_table (id SERIAL PRIMARY KEY)"
            )

        client.grant_all_tables_in_schema(
            grantee=test_grantee,
            privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT],
            schemas=test_schemas,
        )

    def test_grant_all_sequences_multiple_schemas(
        self, client, test_grantee, test_schemas
    ):
        """Test granting sequence privileges across multiple schemas."""
        client.grant_all_sequences_in_schema(
            grantee=test_grantee,
            privileges=[SequencePrivilege.USAGE, SequencePrivilege.SELECT],
            schemas=test_schemas,
        )


# =============================================================================
# End-to-End Permission Setup Test
# =============================================================================


# =============================================================================
# Permission Denied Tests
# =============================================================================


class TestNoRoleUserErrors:
    """Test errors when user has no PostgreSQL role in the Lakebase database.

    REQUIRED ENVIRONMENT VARIABLE:
    ==============================
    Tests in this class require NO_ROLE_USER_TOKEN to be set. This should be an
    OAuth token for a Databricks user who does NOT have a PostgreSQL role created
    in the Lakebase database (i.e., create_role() was never called for them).

    Example:
        export NO_ROLE_USER_TOKEN=<oauth-token-for-user-without-database-role>

    Expected behavior: Connection attempts fail with PoolTimeout. The underlying
    PostgreSQL error is "role does not exist" but the connection pool retries
    until timeout, so tests catch PoolTimeout rather than the raw PostgreSQL error.

    This is different from LIMITED_PERMISSION_USER_TOKEN where the user HAS a role
    but lacks specific GRANT permissions.
    """

    @pytest.fixture
    def no_role_client(self, instance_name):
        """
        Create a client for a user who has no role in the Lakebase database.

        This user can authenticate to Databricks but has never had create_role()
        called for them, so PostgreSQL connection will fail.
        """
        no_role_token = os.environ.get("NO_ROLE_USER_TOKEN")
        if not no_role_token:
            pytest.skip(
                "NO_ROLE_USER_TOKEN not set. "
                "Set this to an OAuth token for a user who has NO role in the Lakebase database. "
                "Example: export NO_ROLE_USER_TOKEN=<oauth-token>"
            )

        workspace_client = create_workspace_client_with_token(no_role_token)
        pool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
        )
        client = LakebaseClient(pool=pool)
        yield client
        client.close()
        pool.close()

    def test_user_without_role_cannot_connect(self, no_role_client, test_grantee, caplog):
        """
        User without a database role should fail to connect OR get permission denied.

        NOTE: Requires NO_ROLE_USER_TOKEN env var. Skipped if not set.

        When a user has no PostgreSQL role in Lakebase, connection attempts fail
        with PoolTimeout. The underlying PostgreSQL error "role does not exist"
        appears in the warning logs during connection retries.

        If the user unexpectedly HAS a role, they should still get a permission
        denied error when trying to grant (since they lack GRANT permissions).
        """
        import logging
        from psycopg_pool import PoolTimeout

        print("\n[INFO] Testing NO_ROLE_USER_TOKEN user attempting to grant schema privileges...")

        # Capture WARNING level logs to verify "role does not exist" message
        with caplog.at_level(logging.WARNING):
            # Try to execute a grant - this should fail either because:
            # 1. User has no role -> PoolTimeout with "role does not exist" in logs
            # 2. User has a role but no GRANT permission -> PermissionError
            try:
                no_role_client.grant_schema(
                    grantee=test_grantee,
                    privileges=[SchemaPrivilege.USAGE],
                    schemas=["public"],
                )
                # If we get here, the grant succeeded - this means the user has
                # both a role AND grant permissions, which is NOT what we expect
                # for a "no role" user. Fail the test.
                pytest.fail(
                    "NO_ROLE_USER_TOKEN user was able to grant privileges! "
                    "This user appears to have a database role with GRANT permissions. "
                    "Please use a token for a user/SP that truly has no role in Lakebase."
                )
            except PoolTimeout as e:
                # Expected: user has no role, connection failed with PoolTimeout
                print(f"[INFO] PoolTimeout raised (expected for no-role user): {e}")

                # Verify the logs contain "role" and "does not exist"
                log_text = " ".join(record.message for record in caplog.records)
                print(f"[INFO] Captured log messages: {log_text[:200]}...")

                assert "role" in log_text.lower(), (
                    f"Expected 'role' in log messages but got: {log_text}"
                )
                assert "does not exist" in log_text.lower(), (
                    f"Expected 'does not exist' in log messages but got: {log_text}"
                )
                print("[INFO] Verified: logs contain 'role' and 'does not exist' messages")

            except PermissionError as e:
                # Also acceptable: user has a role but no GRANT permission
                print(f"[INFO] PermissionError raised (user has role but no GRANT): {e}")
                error_str = str(e)
                assert "Insufficient privileges" in error_str or "permission denied" in error_str.lower()
            except Exception as e:
                # Other connection/permission errors are also acceptable
                # as long as the grant didn't succeed
                print(f"[INFO] Other exception raised: {type(e).__name__}: {e}")
                pass  # Test passes - the grant failed with some error

    def test_create_role_without_database_role_fails(self, no_role_client, caplog):
        """
        User without a database role (or without CAN MANAGE) cannot create roles.

        NOTE: Requires NO_ROLE_USER_TOKEN env var. Skipped if not set.

        The operation should fail either because:
        1. User has no role -> PoolTimeout with "role does not exist" in logs
        2. User has a role but no CAN MANAGE -> PermissionError
        """
        import logging
        from psycopg_pool import PoolTimeout

        print("\n[INFO] Testing NO_ROLE_USER_TOKEN user attempting to create a role...")

        # Capture WARNING level logs to verify "role does not exist" message
        with caplog.at_level(logging.WARNING):
            try:
                no_role_client.create_role(
                    "some-new-sp-uuid",
                    "SERVICE_PRINCIPAL",
                )
                # If we get here, the create succeeded - this is NOT expected. Fail the test.
                pytest.fail(
                    "NO_ROLE_USER_TOKEN user was able to create a role! "
                    "This user appears to have CAN MANAGE permission. "
                    "Please use a token for a user/SP without this permission."
                )
            except PoolTimeout as e:
                print(f"[INFO] PoolTimeout raised (expected for no-role user): {e}")

                # Verify the logs contain "role" and "does not exist"
                log_text = " ".join(record.message for record in caplog.records)
                print(f"[INFO] Captured log messages: {log_text[:200]}...")

                assert "role" in log_text.lower(), (
                    f"Expected 'role' in log messages but got: {log_text}"
                )
                assert "does not exist" in log_text.lower(), (
                    f"Expected 'does not exist' in log messages but got: {log_text}"
                )
                print("[INFO] Verified: logs contain 'role' and 'does not exist' messages")

            except PermissionError as e:
                print(f"[INFO] PermissionError raised (user has role but no CAN MANAGE): {e}")
            except Exception as e:
                print(f"[INFO] Other exception raised: {type(e).__name__}: {e}")


class TestLimitedPermissionUserErrors:
    """Test errors when user has a role but lacks GRANT permissions.

    REQUIRED ENVIRONMENT VARIABLE:
    ==============================
    Tests in this class require LIMITED_PERMISSION_USER_TOKEN to be set. This should
    be an OAuth token for a Databricks user who:
    1. HAS a PostgreSQL role in the Lakebase database (create_role was called for them)
    2. Does NOT have 'CAN MANAGE' permission on the Lakebase instance
    3. Does NOT have GRANT OPTION on schemas/tables

    Example:
        export LIMITED_PERMISSION_USER_TOKEN=<oauth-token-for-user-with-role-but-no-grant>

    This user can connect to the database but cannot grant privileges to others.

    To set up this user:
    1. As an admin, call: client.create_role("user@example.com", "USER")
    2. Grant them basic access: GRANT USAGE ON SCHEMA public TO "user@example.com"
    3. Do NOT give them 'CAN MANAGE' on the Lakebase instance
    """

    @pytest.fixture
    def limited_permission_client(self, instance_name):
        """
        Create a client for a user who has a role but limited permissions.

        This user can connect to PostgreSQL but cannot grant privileges to others
        because they lack 'CAN MANAGE' and GRANT OPTION.
        """
        limited_token = os.environ.get("LIMITED_PERMISSION_USER_TOKEN")
        if not limited_token:
            pytest.skip(
                "LIMITED_PERMISSION_USER_TOKEN not set. "
                "Set this to an OAuth token for a user who HAS a database role "
                "but lacks GRANT permissions. "
                "Example: export LIMITED_PERMISSION_USER_TOKEN=<oauth-token>"
            )

        workspace_client = create_workspace_client_with_token(limited_token)
        pool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
        )
        client = LakebaseClient(pool=pool)
        yield client
        client.close()
        pool.close()

    def test_grant_schema_without_permission_raises_error(
        self, limited_permission_client, test_grantee
    ):
        """
        Granting schema privileges without proper permissions should raise PermissionError.

        NOTE: Requires LIMITED_PERMISSION_USER_TOKEN env var. Skipped if not set.

        SETUP REQUIRED: Create 'test_limited_perm_schema' as admin before running:
            CREATE SCHEMA IF NOT EXISTS test_limited_perm_schema;

        Uses a dedicated test schema (not 'public') because PostgreSQL's public schema
        often has special default permissions that allow broader GRANT access.
        """
        test_schema = "test_limited_perm_schema"
        print(f"\n[INFO] Attempting to grant on '{test_schema}' with limited permission user...")

        try:
            limited_permission_client.grant_schema(
                grantee=test_grantee,
                privileges=[SchemaPrivilege.USAGE],
                schemas=[test_schema],
            )
            pytest.fail(
                f"LIMITED_PERMISSION_USER_TOKEN user was able to grant on '{test_schema}'! "
                "This user should NOT have GRANT permission. "
                "Ensure this schema is owned by a different user."
            )
        except PermissionError as e:
            print(f"[INFO] grant_schema raised PermissionError (expected): {e}")
            error_msg = str(e)
            assert "Insufficient privileges" in error_msg or "permission denied" in error_msg.lower()
        except ValueError as e:
            # Schema might not exist
            if "does not exist" in str(e):
                pytest.fail(
                    f"Schema '{test_schema}' does not exist. "
                    f"Create it as admin: CREATE SCHEMA IF NOT EXISTS {test_schema};"
                )
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected exception: {type(e).__name__}: {e}")
            raise

    def test_grant_table_without_permission_raises_error(
        self, limited_permission_client, test_grantee
    ):
        """
        Granting table privileges without proper permissions should raise PermissionError.

        NOTE: Requires LIMITED_PERMISSION_USER_TOKEN env var. Skipped if not set.
        """
        try:
            limited_permission_client.grant_all_tables_in_schema(
                grantee=test_grantee,
                privileges=[TablePrivilege.SELECT],
                schemas=["public"],
            )
            pytest.fail(
                "LIMITED_PERMISSION_USER_TOKEN user was able to grant table privileges! "
                "This user should NOT have GRANT permission on tables."
            )
        except PermissionError as e:
            print(f"\n[INFO] grant_all_tables_in_schema raised PermissionError (expected): {e}")
            error_msg = str(e)
            assert "Insufficient privileges" in error_msg or "permission denied" in error_msg.lower()
        except Exception as e:
            print(f"\n[ERROR] grant_all_tables_in_schema raised unexpected exception: {type(e).__name__}")
            print(f"[ERROR] Exception message: {e}")
            raise

    def test_create_role_without_can_manage_raises_error(self, limited_permission_client):
        """
        Creating a role without 'CAN MANAGE' permission should raise PermissionError.

        NOTE: Requires LIMITED_PERMISSION_USER_TOKEN env var. Skipped if not set.

        Expected error message should include:
        - What operation failed
        - Suggestion to check 'CAN MANAGE' permission
        """
        try:
            limited_permission_client.create_role(
                "some-new-sp-uuid",
                "SERVICE_PRINCIPAL",
            )
            pytest.fail(
                "LIMITED_PERMISSION_USER_TOKEN user was able to create a role! "
                "This user should NOT have CAN MANAGE permission."
            )
        except PermissionError as e:
            print(f"\n[INFO] create_role raised PermissionError (expected): {e}")
            error_msg = str(e)
            assert "Insufficient privileges" in error_msg or "permission denied" in error_msg.lower()
            assert "CAN MANAGE" in error_msg
        except Exception as e:
            print(f"\n[ERROR] create_role raised unexpected exception: {type(e).__name__}")
            print(f"[ERROR] Exception message: {e}")
            raise


class TestObjectNotFoundErrors:
    """Test errors when granting on non-existent objects.

    These tests use the privileged client (DATABRICKS_TOKEN) to verify that
    proper ValueError is raised when objects don't exist.
    """

    def test_grant_on_nonexistent_schema_raises_error(self, client, test_grantee):
        """
        Granting privileges on a non-existent schema should raise ValueError.

        Expected error message should include:
        - What operation failed
        - That the object doesn't exist
        - Suggestion to verify schema exists
        """
        with pytest.raises((ValueError, PermissionError)) as exc_info:
            client.grant_schema(
                grantee=test_grantee,
                privileges=[SchemaPrivilege.USAGE],
                schemas=["nonexistent_schema_xyz_12345"],
            )

        error_msg = str(exc_info.value)
        assert "does not exist" in error_msg or "not exist" in error_msg.lower()

    def test_grant_on_nonexistent_table_raises_error(self, client, test_grantee):
        """
        Granting privileges on a non-existent table should raise ValueError.
        """
        with pytest.raises((ValueError, PermissionError)) as exc_info:
            client.grant_table(
                grantee=test_grantee,
                privileges=[TablePrivilege.SELECT],
                tables=["public.nonexistent_table_xyz_12345"],
            )

        error_msg = str(exc_info.value)
        assert "does not exist" in error_msg or "not exist" in error_msg.lower()

    def test_grant_to_nonexistent_role_raises_error(self, client):
        """
        Granting privileges to a non-existent role should raise ValueError.

        The role must be created with create_role() before granting privileges.
        """
        with pytest.raises((ValueError, PermissionError)) as exc_info:
            client.grant_schema(
                grantee="nonexistent_role_xyz_12345",
                privileges=[SchemaPrivilege.USAGE],
                schemas=["public"],
            )

        error_msg = str(exc_info.value)
        assert "does not exist" in error_msg or "not exist" in error_msg.lower()


class TestEndToEndPermissionSetup:
    """Test complete permission setup workflow."""

    def test_full_permission_setup_for_app(self, client, test_grantee):
        """
        Test a complete permission setup workflow for an application.

        This represents a realistic use case where an app needs:
        1. Role creation (handled by test_grantee fixture)
        2. Schema access
        3. Table CRUD permissions
        4. Sequence permissions for auto-increment columns
        """
        # 1. Role already created by test_grantee fixture

        # 2. Grant schema privileges
        client.grant_schema(
            grantee=test_grantee,
            privileges=[SchemaPrivilege.USAGE],
            schemas=["public"],
        )

        # 3. Grant table privileges
        client.grant_all_tables_in_schema(
            grantee=test_grantee,
            privileges=[
                TablePrivilege.SELECT,
                TablePrivilege.INSERT,
                TablePrivilege.UPDATE,
                TablePrivilege.DELETE,
            ],
            schemas=["public"],
        )

        # 4. Grant sequence privileges (needed for INSERT with SERIAL columns)
        client.grant_all_sequences_in_schema(
            grantee=test_grantee,
            privileges=[
                SequencePrivilege.USAGE,
                SequencePrivilege.SELECT,
                SequencePrivilege.UPDATE,
            ],
            schemas=["public"],
        )
