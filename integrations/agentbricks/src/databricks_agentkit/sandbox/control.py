"""Invocation state and UI snapshots; source publications are Workspace files."""

import hashlib
import json
import logging
import threading
import time
from contextlib import closing, contextmanager, nullcontext
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from databricks_agentkit.sandbox.bundle import Bundle
from databricks_agentkit.sandbox.catalog import WorkspaceCatalog

RUN_SELECT = "SELECT * FROM invocations "
STREAM_FLUSH_SECONDS = 0.05
STREAM_RESYNC_SECONDS = 5
LOG = logging.getLogger(__name__)


def run_channel(namespace, run_id):
    # PostgreSQL identifiers are at most 63 bytes. Do not expose session text.
    return "ab_run_" + hashlib.sha256(f"{namespace}\0{run_id}".encode()).hexdigest()[:48]


def publish_update(db, namespace, run_id, statement, params):
    """Commit a fenced update and its wakeup in one statement/transaction.

    Notifications carry no text. Readers always load the committed snapshot,
    making coalesced notifications and reconnects safe.
    """
    return db.execute(
        "WITH changed AS (" + statement + " RETURNING 1) SELECT pg_notify(%s, '') FROM changed",
        (*params, run_channel(namespace, run_id)),
    )


def notify_run(db, namespace, run_id):
    db.execute("SELECT pg_notify(%s, '')", (run_channel(namespace, run_id),))


class ControlStore:
    def __init__(self, registry, catalog=None):
        self.registry = registry
        self.namespace = registry.namespace
        self.catalog = catalog or WorkspaceCatalog(
            registry.workspace, f"/Shared/agentbricks-runtime/{self.namespace}"
        )

    def initialize(self):
        with self.registry.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS invocations (
                    namespace TEXT NOT NULL, id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, invocation_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, input JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', stage TEXT NOT NULL DEFAULT 'Queued',
                    result JSONB, error TEXT, lifecycle JSONB, attempt_id TEXT, stream JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (namespace, id),
                    UNIQUE(namespace, agent_id, session_id, invocation_id)
                )
            """)

    def create_agent(
        self,
        name,
        kind,
        path,
        entrypoint,
        bundle: Bundle,
        *,
        git_source=None,
        execution_artifact=None,
    ):
        return self.catalog.create_agent(
            name,
            kind,
            path,
            entrypoint,
            bundle,
            git_source=git_source,
            execution_artifact=execution_artifact,
        )

    def agents(self):
        return self.catalog.list("revisions")

    def agent(self, agent_id, *, with_bundle=False):
        return self.catalog.agent(agent_id, with_bundle=with_bundle)

    def create_run(
        self, agent_id, session_id, invocation_id, value, *, retry=False, connection=None
    ):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        # The immutable revision is part of the invocation identity; no alias join.
        with nullcontext(connection) if connection is not None else self.registry.connect() as db:
            assert db is not None
            if (
                retry
                and not db.execute(
                    "SELECT 1 FROM invocations WHERE namespace=%s AND agent_id=%s "
                    "AND session_id=%s AND invocation_id=%s",
                    (self.namespace, agent_id, session_id, invocation_id),
                ).fetchone()
            ):
                raise KeyError("Original invocation not found; submit without retry")
            db.execute(
                "INSERT INTO invocations "
                "(namespace,id,agent_id,session_id,invocation_id,request_hash,input) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT "
                "(namespace,agent_id,session_id,invocation_id) DO NOTHING",
                (
                    self.namespace,
                    uuid4().hex,
                    agent_id,
                    session_id,
                    invocation_id,
                    digest,
                    Jsonb(value),
                ),
            )
            row = db.execute(
                "SELECT * FROM invocations WHERE "
                "namespace=%s AND agent_id=%s AND session_id=%s AND invocation_id=%s",
                (self.namespace, agent_id, session_id, invocation_id),
            ).fetchone()
            if row["request_hash"] != digest:
                raise ValueError("Invocation ID already used with different input")
            if retry and row["status"] in {"failed", "interrupted"}:
                row = (
                    db.execute(
                        "UPDATE invocations SET status='queued', stage='Queued', "
                        "error=NULL, result=NULL, lifecycle=NULL, attempt_id=NULL, stream=NULL, "
                        "updated_at=now() WHERE namespace=%s AND id=%s "
                        "AND status IN ('failed','interrupted') RETURNING *",
                        (self.namespace, row["id"]),
                    ).fetchone()
                    or row
                )
                notify_run(db, self.namespace, row["id"])
            return row

    def run(self, run_id, *, connection=None):
        with nullcontext(connection) if connection is not None else self.registry.connect() as db:
            assert db is not None
            row = db.execute(
                RUN_SELECT + "WHERE namespace=%s AND id=%s",
                (self.namespace, run_id),
            ).fetchone()
        if row is None:
            raise KeyError("Invocation not found")
        return row

    def claim(self, run_id):
        with self.registry.connect() as db, db.transaction():
            row = db.execute(
                "UPDATE invocations SET status='running', stage='Starting', "
                "lifecycle=NULL, stream=NULL, attempt_id=%s, "
                "updated_at=now() WHERE namespace=%s AND id=%s AND status='queued' RETURNING *",
                (uuid4().hex, self.namespace, run_id),
            ).fetchone()
            if row is not None:
                notify_run(db, self.namespace, run_id)
            return row

    @contextmanager
    def stream(self, run_id, *, attempt_id=None):
        with self.registry.connect() as db:
            # Dedicated preview connection; bound SQL work before joining its writer.
            db.execute("SET statement_timeout = '5s'")
            attempt = uuid4().hex
            publish_update(
                db,
                self.namespace,
                run_id,
                "UPDATE invocations SET stream=%s WHERE namespace=%s AND id=%s AND status='running'"
                + (" AND attempt_id=%s" if attempt_id is not None else ""),
                (
                    Jsonb({"attempt": attempt, "sequence": 0, "text": "", "first_token_ms": None}),
                    self.namespace,
                    run_id,
                )
                + ((attempt_id,) if attempt_id is not None else ()),
            )
            with PartialText(db, self.namespace, run_id, attempt) as stream:
                yield stream.emit

    def watch(self, run_id):
        # LISTEN must commit before the first read: a write between subscribing
        # and reading is either in that snapshot or queued as a notification.
        with self.registry.connect() as db:
            db.execute(
                sql.SQL("LISTEN {}").format(sql.Identifier(run_channel(self.namespace, run_id)))
            )
            while True:
                row = self.run(run_id, connection=db)
                yield row
                if row["status"] not in {"queued", "running"}:
                    return
                # A quiet-stream resync/heartbeat recovers missed wakeups. Normal
                # delivery is notification-driven, including across App processes.
                # Exhaust/close before querying: notifies holds the connection lock.
                with closing(db.notifies(timeout=STREAM_RESYNC_SECONDS, stop_after=1)) as notices:
                    for _ in notices:
                        pass

    def progress(self, run_id, stage, *, attempt_id=None):
        with self.registry.connect() as db:
            publish_update(
                db,
                self.namespace,
                run_id,
                "UPDATE invocations SET stage=%s, updated_at=now() "
                "WHERE namespace=%s AND id=%s AND status='running'"
                + (" AND attempt_id=%s" if attempt_id is not None else ""),
                (stage, self.namespace, run_id) + ((attempt_id,) if attempt_id is not None else ()),
            )

    @contextmanager
    def track(self, run_id, attempt_id):
        # Keep one connection for phase updates; only transitions write to Lakebase.
        # The token fences delayed updates from a previous attempt of the same run.
        with self.registry.connect() as db:

            def publish(snapshot):
                publish_update(
                    db,
                    self.namespace,
                    run_id,
                    "UPDATE invocations SET lifecycle=%s, updated_at=now() "
                    "WHERE namespace=%s AND id=%s AND attempt_id=%s AND status='running'",
                    (Jsonb(snapshot), self.namespace, run_id, attempt_id),
                )

            yield publish

    def finish(self, run_id, *, result=None, error=None, attempt_id=None):
        status = "failed" if error else "succeeded"
        with self.registry.connect() as db:
            publish_update(
                db,
                self.namespace,
                run_id,
                "UPDATE invocations SET status=%s, stage=%s, result=%s, "
                "error=%s, updated_at=now() WHERE namespace=%s AND id=%s AND status='running'"
                + (" AND attempt_id=%s" if attempt_id is not None else ""),
                (
                    status,
                    "Failed" if error else "Complete",
                    Jsonb(result),
                    error,
                    self.namespace,
                    run_id,
                )
                + ((attempt_id,) if attempt_id is not None else ()),
            )

    def recover(self):
        # One App process owns this namespace. Don't automatically replay interrupted
        # side effects; explicit retry keeps the original invocation ID.
        with self.registry.connect() as db, db.transaction():
            interrupted = db.execute(
                "UPDATE invocations SET status='interrupted', stage='Interrupted', "
                "error='App restarted during execution; retry explicitly with the same invocation ID', "
                "updated_at=now() WHERE namespace=%s AND status='running' RETURNING id",
                (self.namespace,),
            ).fetchall()
            for row in interrupted:
                notify_run(db, self.namespace, row["id"])
            return db.execute(
                "SELECT id FROM invocations WHERE namespace=%s AND status='queued' ORDER BY created_at",
                (self.namespace,),
            ).fetchall()

    def sessions(self, agent_id):
        with self.registry.connect() as db:
            return db.execute(
                "SELECT * FROM (SELECT DISTINCT ON (session_id) session_id, id AS run_id, status, stage, "
                "created_at, updated_at, result->>'sandbox_id' AS sandbox_id "
                "FROM invocations WHERE namespace=%s AND agent_id=%s "
                "ORDER BY session_id, updated_at DESC) latest ORDER BY updated_at DESC LIMIT 100",
                (self.namespace, agent_id),
            ).fetchall()

    def history(self, agent_id, session_id):
        with self.registry.connect() as db:
            return db.execute(
                "SELECT * FROM ("
                + RUN_SELECT
                + "WHERE namespace=%s AND agent_id=%s AND session_id=%s "
                "ORDER BY created_at DESC LIMIT 100) history ORDER BY created_at",
                (self.namespace, agent_id, session_id),
            ).fetchall()


def scoped_session(agent_id: str, session_id: str) -> str:
    return f"agent:{agent_id}\0{session_id}"


class PartialText:
    """One bounded buffer and timer-driven writer per active invocation.

    The tail-reader callback never waits on database I/O. A quiet final fragment
    is flushed on the same deadline as active output, independently of handler
    completion. Slow writes coalesce text rather than building an unbounded queue.
    """

    def __init__(self, db, namespace, run_id, attempt, *, interval=STREAM_FLUSH_SECONDS):
        self.db, self.namespace, self.run_id, self.attempt = db, namespace, run_id, attempt
        self.interval = interval
        self.started = time.monotonic()
        self.text, self.first_token_ms, self.dirty = "", None, False
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._write, name="sandbox-text-writer", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def emit(self, delta):
        with self.condition:
            if self.closed or self.error or not delta or len(self.text) >= 256 * 1024:
                return
            if self.first_token_ms is None:
                self.first_token_ms = round((time.monotonic() - self.started) * 1000, 2)
            self.text += delta[: 256 * 1024 - len(self.text)]
            self.dirty = True
            self.condition.notify()

    def _write(self):
        flushed = None
        try:
            while True:
                with self.condition:
                    while not self.dirty and not self.closed:
                        self.condition.wait()
                    if not self.dirty:
                        return
                    if not self.closed and flushed is not None:
                        remaining = flushed + self.interval - time.monotonic()
                        if remaining > 0:
                            self.condition.wait(remaining)
                            continue
                    text, first_token_ms = self.text, self.first_token_ms
                    self.dirty = False
                publish_update(
                    self.db,
                    self.namespace,
                    self.run_id,
                    "UPDATE invocations SET stream=stream || jsonb_build_object("
                    "'text',%s::text,'sequence',(stream->>'sequence')::integer+1,"
                    "'first_token_ms',%s::double precision) "
                    "WHERE namespace=%s AND id=%s AND stream->>'attempt'=%s AND status='running'",
                    (text, first_token_ms, self.namespace, self.run_id, self.attempt),
                )
                flushed = time.monotonic()
        except Exception as exc:
            with self.condition:
                self.error = exc
            # Preview failure cannot turn completed agent side effects into a
            # retryable invocation failure. The final result remains authoritative.
            LOG.warning(
                "Live text publication failed; final result will carry the answer", exc_info=True
            )

    def __exit__(self, *exc):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            # SQL timeout bounds server work; close also interrupts a broken socket.
            # This connection belongs exclusively to this preview writer.
            self.db.close()
            self.thread.join(timeout=1)
            LOG.warning("Live text publication timed out during shutdown")
