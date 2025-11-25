from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Optional

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.store.base import BaseStore, Item, Op, Result
    from langgraph.store.postgres import PostgresStore

    _store_imports_available = True
except ImportError:
    LakebasePool = object
    PostgresStore = object
    BaseStore = object
    Item = object
    Op = object
    Result = object
    _store_imports_available = False


class DatabricksStore(BaseStore):
    """Provides APIs for working with long-term memory on Databricks using Lakebase.
    Extends LangGraph BaseStore interface using Databricks Lakebase for connection pooling.

    Operations borrow a connection from the pool, create a short-lived PostgresStore,
    execute the operation, and return the connection to the pool.
    """

    @staticmethod
    def normalize_namespace_label(s: Optional[str]) -> str:
        """Normalize a string for use as a namespace label.
        Converts to lowercase, replaces @ with -at-, removes invalid characters,
        and truncates with hash if too long.
        Args:
            s: The string to normalize (e.g., email address, user_id)
        Returns:
            Normalized string safe for namespace usage
        Example:
            >>> normalize_namespace_label("user@example.com")
            'user-at-example-com'
            >>> normalize_namespace_label("")
            'anon'
        """
        SAFE_NS_MAX = 64

        if not s:
            return "anon"
        x = s.strip().lower().replace("@", "-at-")
        x = re.sub(r"[^a-z0-9_-]+", "-", x)  # removes dots and punctuation
        x = re.sub(r"-{2,}", "-", x).strip("-") or "anon"
        if len(x) > SAFE_NS_MAX:
            head = x[: SAFE_NS_MAX - 17]
            tail = hashlib.sha256(x.encode()).hexdigest()[:16]
            x = f"{head}-{tail}"
        return x

    @staticmethod
    def namespace(identifier: str, prefix: str = "users") -> tuple[str, ...]:
        """Create a namespace tuple with a normalized identifier.
        Args:
            identifier: The identifier to normalize (e.g., user_id, email, entity_name)
            prefix: The namespace prefix (default: "users")
        Returns:
            Tuple of (prefix, normalized_identifier) for use as namespace
        Example:
            >>> namespace("email@databricks.com")
            ('users', 'email-at-databricks-com')
            >>> namespace("session-123", prefix="sessions")
            ('sessions', 'session-123')
        """
        return (prefix, DatabricksStore.normalize_namespace_label(identifier))

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        **pool_kwargs: Any,
    ) -> None:
        if not _store_imports_available:
            raise ImportError(
                "DatabricksStore requires databricks-langchain[memory]. "
                "Install with: pip install 'databricks-langchain[memory]'"
            )

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **pool_kwargs,
        )

    def _with_store(self, fn, *args, **kwargs):
        """
        Borrow a connection, create a short-lived PostgresStore, call fn(store),
        then return the connection to the pool.
        """
        with self._lakebase.connection() as conn:
            store = PostgresStore(conn=conn)
            return fn(store, *args, **kwargs)

    def setup(self) -> None:
        """Instantiate the store, setting up necessary persistent storage."""
        return self._with_store(lambda s: s.setup())

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations synchronously.

        This is the core method required by BaseStore. All other operations
        (get, put, search, delete, list_namespaces) are inherited from BaseStore
        and internally call this batch() method.
        """
        return self._with_store(lambda s: s.batch(ops))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        This is the second abstract method required by BaseStore.
        Currently delegates to sync batch() - for true async support,
        would need async-compatible connection pooling.
        """
        return self.batch(ops)
