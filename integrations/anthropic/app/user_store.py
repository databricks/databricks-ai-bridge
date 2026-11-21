"""State for user mode (M2): which Databricks user owns which Claude session, sign-in codes, and an audit trail.

Two implementations behind one small interface:

- :class:`LakebaseStore` (production): Lakebase Postgres, reached as the App's service principal. The App owns
  schema ``cma_dispatcher`` (created on first use, never ``public``). Bindings and the audit trail are
  append-only: nothing here updates or deletes them. The only deletes purge sign-in codes (and their use rows) a
  day after the code expired; an expired code can't be redeemed, so purging it can't make a code usable again. A
  sign-in code is single-use because redeeming it inserts its hash into ``connect_uses`` (primary key): a second
  redeem finds the row taken.
- :class:`MemoryStore` (tests and local runs): the same behaviour in a dict.

Nothing here stores a forwarded user token or a secret value; sign-in codes are stored as SHA-256 hashes.
Rows are keyed by the user's immutable Databricks id (``principal_id``); ``user_name`` is kept for display and
for the session's metadata only.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from databricks.sdk import WorkspaceClient

log = logging.getLogger("cma_dispatcher_app.store")

SCHEMA = "cma_dispatcher"
# Lakebase credentials last one hour: refresh well before that.
PASSWORD_TTL_S = 50 * 60
MAX_LIST = 200

DDL = (
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.bindings (
        session_id text PRIMARY KEY,
        principal_id text NOT NULL,
        user_name text NOT NULL,
        agent_id text NOT NULL,
        created_at timestamptz DEFAULT now())""",
    f"CREATE INDEX IF NOT EXISTS bindings_principal ON {SCHEMA}.bindings (principal_id)",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.audit (
        id bigserial PRIMARY KEY,
        at timestamptz DEFAULT now(),
        principal_id text,
        session_id text,
        action text NOT NULL,
        detail text)""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.connect_codes (
        code_hash text PRIMARY KEY,
        session_id text NOT NULL,
        created_at timestamptz DEFAULT now(),
        expires_at timestamptz NOT NULL)""",
    f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.connect_uses (
        code_hash text PRIMARY KEY,
        principal_id text NOT NULL,
        used_at timestamptz DEFAULT now())""",
)


class StoreUnavailable(RuntimeError):
    """The state store can't be reached or failed. The message never contains credentials."""


@dataclass(frozen=True)
class Binding:
    session_id: str
    principal_id: str
    user_name: str
    agent_id: str
    created_at: str = ""


# Sign-in codes are purged a day after they expire (an hourly sweep in user_mode.py).
PURGE_GRACE_S = 24 * 3600.0


class Store(Protocol):
    def ensure_schema(self) -> None: ...

    def bind(self, binding: Binding) -> None: ...

    def binding(self, session_id: str) -> Binding | None: ...

    def bindings_for(self, principal_id: str) -> list[Binding]: ...

    def count_for(self, principal_id: str) -> int: ...

    def audit(
        self, principal_id: str, session_id: str | None, action: str, detail: str = ""
    ) -> None: ...

    def add_code(self, code_hash: str, session_id: str, ttl_s: float) -> None: ...

    def peek_code(self, code_hash: str) -> str | None: ...

    def use_code(self, code_hash: str, principal_id: str) -> str | None: ...

    def purge_expired_codes(self, grace_s: float = PURGE_GRACE_S) -> int: ...


class MemoryStore:
    """In-process store with the same semantics (unique session ids, insert-only audit)."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._bindings: dict[str, Binding] = {}
        self._codes: dict[str, tuple[str, float]] = {}  # code hash -> (session id, expires at)
        self._uses: dict[str, str] = {}  # code hash -> principal id
        self.audit_rows: list[tuple[str, str | None, str, str]] = []

    def ensure_schema(self) -> None:
        return None

    def bind(self, binding: Binding) -> None:
        with self._lock:
            if binding.session_id in self._bindings:
                raise StoreUnavailable("binding already exists")
            self._bindings[binding.session_id] = binding

    def binding(self, session_id: str) -> Binding | None:
        with self._lock:
            return self._bindings.get(session_id)

    def bindings_for(self, principal_id: str) -> list[Binding]:
        with self._lock:
            rows = [b for b in self._bindings.values() if b.principal_id == principal_id]
        return rows[::-1][:MAX_LIST]

    def count_for(self, principal_id: str) -> int:
        with self._lock:
            return sum(1 for b in self._bindings.values() if b.principal_id == principal_id)

    def audit(
        self, principal_id: str, session_id: str | None, action: str, detail: str = ""
    ) -> None:
        with self._lock:
            self.audit_rows.append((principal_id, session_id, action, detail))

    def add_code(self, code_hash: str, session_id: str, ttl_s: float) -> None:
        with self._lock:
            if code_hash in self._codes:
                raise StoreUnavailable("code already exists")
            self._codes[code_hash] = (session_id, self._clock() + ttl_s)

    def peek_code(self, code_hash: str) -> str | None:
        with self._lock:
            hit = self._codes.get(code_hash)
            if not hit or hit[1] <= self._clock() or code_hash in self._uses:
                return None
            return hit[0]

    def use_code(self, code_hash: str, principal_id: str) -> str | None:
        with self._lock:
            hit = self._codes.get(code_hash)
            if not hit or hit[1] <= self._clock() or code_hash in self._uses:
                return None
            self._uses[code_hash] = principal_id
            return hit[0]

    def purge_expired_codes(self, grace_s: float = PURGE_GRACE_S) -> int:
        """Drop codes that expired more than ``grace_s`` ago, and their use rows. Returns codes dropped."""
        with self._lock:
            cutoff = self._clock() - grace_s
            gone = [h for h, (_, expires) in self._codes.items() if expires < cutoff]
            for h in gone:
                del self._codes[h]
                self._uses.pop(h, None)
            return len(gone)


class UnavailableStore:
    """Stands in when Lakebase isn't attached: every call raises StoreUnavailable, so user routes answer 503."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def _fail(self, *args: Any, **kwargs: Any) -> Any:
        raise StoreUnavailable(self.reason)

    bind = binding = bindings_for = count_for = audit = ensure_schema = _fail
    add_code = peek_code = use_code = purge_expired_codes = _fail


class LakebaseStore:
    """Lakebase Postgres as the App's service principal (psycopg 3, one short connection per operation).

    The connection settings come from the variables the ``postgres`` App resource injects (``PGHOST``,
    ``PGPORT``, ``PGDATABASE``, ``PGUSER``, ``PGSSLMODE``) and ``LAKEBASE_ENDPOINT`` (set by the bundle).
    The password is a short-lived OAuth credential generated with the App's own identity and refreshed
    before it expires.
    """

    def __init__(
        self,
        *,
        host: str,
        port: str,
        database: str,
        user: str,
        sslmode: str,
        endpoint: str,
        workspace_client: WorkspaceClient,
        connect: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host, self._port, self._database = host, port, database
        self._user, self._sslmode, self._endpoint = user, sslmode, endpoint
        self._w = workspace_client
        self._connect = connect  # tests inject a fake; production imports psycopg on first use
        self._clock = clock
        self._password: str | None = None
        self._password_at = 0.0
        self._schema_ready = False
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, env: dict[str, str], workspace_client: WorkspaceClient) -> Store:
        """The store for this App's environment; :class:`UnavailableStore` if Lakebase isn't attached."""
        missing = [
            k for k in ("PGHOST", "PGDATABASE", "PGUSER", "LAKEBASE_ENDPOINT") if not env.get(k)
        ]
        if missing:
            return UnavailableStore(f"Lakebase not attached (missing {', '.join(missing)})")
        return cls(
            host=env["PGHOST"],
            port=env.get("PGPORT") or "5432",
            database=env["PGDATABASE"],
            user=env["PGUSER"],
            sslmode=env.get("PGSSLMODE") or "require",
            endpoint=env["LAKEBASE_ENDPOINT"],
            workspace_client=workspace_client,
        )

    def _credential(self) -> str:
        with self._lock:
            fresh = self._password and self._clock() - self._password_at < PASSWORD_TTL_S
            if not fresh:
                self._password = self._w.postgres.generate_database_credential(
                    endpoint=self._endpoint
                ).token
                self._password_at = self._clock()
            return self._password  # type: ignore[return-value]

    def _open(self):
        connect = self._connect
        if connect is None:
            connect = importlib.import_module("psycopg").connect
        try:
            conn = connect(
                host=self._host,
                port=self._port,
                dbname=self._database,
                user=self._user,
                password=self._credential(),
                sslmode=self._sslmode,
                connect_timeout=5,
                application_name="cma-dispatcher",
                autocommit=True,
            )
        except Exception as e:  # noqa: BLE001 - report the type only; the message may echo the host
            with self._lock:
                self._password = None  # a rejected credential must not be reused
            raise StoreUnavailable(f"cannot connect to Lakebase ({type(e).__name__})") from None
        if not self._schema_ready:
            try:
                for statement in DDL:
                    conn.execute(statement)
            except Exception as e:  # noqa: BLE001
                conn.close()
                raise StoreUnavailable(f"cannot prepare schema ({type(e).__name__})") from None
            self._schema_ready = True
        return conn

    def _run(self, fn: Callable[[Any], Any]) -> Any:
        try:
            with self._open() as conn:
                return fn(conn)
        except StoreUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise StoreUnavailable(f"Lakebase query failed ({type(e).__name__})") from None

    def ensure_schema(self) -> None:
        self._run(lambda conn: None)

    def bind(self, binding: Binding) -> None:
        self._run(
            lambda conn: conn.execute(
                f"INSERT INTO {SCHEMA}.bindings (session_id, principal_id, user_name, agent_id) "
                "VALUES (%s, %s, %s, %s)",
                (binding.session_id, binding.principal_id, binding.user_name, binding.agent_id),
            )
        )

    def binding(self, session_id: str) -> Binding | None:
        rows = self._run(
            lambda conn: conn.execute(
                f"SELECT session_id, principal_id, user_name, agent_id, created_at "
                f"FROM {SCHEMA}.bindings WHERE session_id = %s",
                (session_id,),
            ).fetchall()
        )
        return _binding(rows[0]) if rows else None

    def bindings_for(self, principal_id: str) -> list[Binding]:
        rows = self._run(
            lambda conn: conn.execute(
                f"SELECT session_id, principal_id, user_name, agent_id, created_at "
                f"FROM {SCHEMA}.bindings WHERE principal_id = %s ORDER BY created_at DESC LIMIT %s",
                (principal_id, MAX_LIST),
            ).fetchall()
        )
        return [_binding(r) for r in rows]

    def count_for(self, principal_id: str) -> int:
        rows = self._run(
            lambda conn: conn.execute(
                f"SELECT count(*) FROM {SCHEMA}.bindings WHERE principal_id = %s", (principal_id,)
            ).fetchall()
        )
        return int(rows[0][0])

    def audit(
        self, principal_id: str, session_id: str | None, action: str, detail: str = ""
    ) -> None:
        # Append-only: the audit trail is never updated or deleted (a unit test checks the SQL).
        self._run(
            lambda conn: conn.execute(
                f"INSERT INTO {SCHEMA}.audit (principal_id, session_id, action, detail) "
                "VALUES (%s, %s, %s, %s)",
                (principal_id, session_id, action, detail),
            )
        )

    def add_code(self, code_hash: str, session_id: str, ttl_s: float) -> None:
        self._run(
            lambda conn: conn.execute(
                f"INSERT INTO {SCHEMA}.connect_codes (code_hash, session_id, expires_at) "
                "VALUES (%s, %s, now() + make_interval(secs => %s))",
                (code_hash, session_id, float(ttl_s)),
            )
        )

    def peek_code(self, code_hash: str) -> str | None:
        rows = self._run(
            lambda conn: conn.execute(
                f"SELECT c.session_id FROM {SCHEMA}.connect_codes c "
                f"WHERE c.code_hash = %s AND c.expires_at > now() "
                f"AND NOT EXISTS (SELECT 1 FROM {SCHEMA}.connect_uses u WHERE u.code_hash = c.code_hash)",
                (code_hash,),
            ).fetchall()
        )
        return rows[0][0] if rows else None

    def use_code(self, code_hash: str, principal_id: str) -> str | None:
        """Redeem a code once: the insert into ``connect_uses`` (primary key) is what makes it single-use."""

        def redeem(conn: Any) -> str | None:
            rows = conn.execute(
                f"SELECT session_id FROM {SCHEMA}.connect_codes WHERE code_hash = %s AND expires_at > now()",
                (code_hash,),
            ).fetchall()
            if not rows:
                return None
            taken = conn.execute(
                f"INSERT INTO {SCHEMA}.connect_uses (code_hash, principal_id) VALUES (%s, %s) "
                "ON CONFLICT (code_hash) DO NOTHING RETURNING code_hash",
                (code_hash, principal_id),
            ).fetchall()
            return rows[0][0] if taken else None

        return self._run(redeem)

    def purge_expired_codes(self, grace_s: float = PURGE_GRACE_S) -> int:
        """Delete sign-in codes that expired more than ``grace_s`` ago, then use rows whose code is gone.

        The only deletes in this store. Bindings and the audit trail are never deleted or changed.
        """

        def purge(conn: Any) -> int:
            gone = conn.execute(
                f"DELETE FROM {SCHEMA}.connect_codes "
                "WHERE expires_at < now() - make_interval(secs => %s) RETURNING code_hash",
                (float(grace_s),),
            ).fetchall()
            conn.execute(
                f"DELETE FROM {SCHEMA}.connect_uses u "
                f"WHERE NOT EXISTS (SELECT 1 FROM {SCHEMA}.connect_codes c WHERE c.code_hash = u.code_hash)"
            )
            return len(gone)

        return self._run(purge)


def _binding(row: tuple) -> Binding:
    session_id, principal_id, user_name, agent_id, created_at = row
    return Binding(
        session_id=session_id,
        principal_id=principal_id,
        user_name=user_name,
        agent_id=agent_id,
        created_at=created_at.isoformat() if created_at else "",
    )
