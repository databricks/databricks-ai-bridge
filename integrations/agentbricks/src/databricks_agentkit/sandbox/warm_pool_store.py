"""Durable, deployment-scoped inventory for unassigned prepared sandboxes."""

from uuid import uuid4

from psycopg.types.json import Jsonb


class WarmPoolStore:
    def __init__(self, registry, pool_key, generation, size):
        self.registry, self.key = registry, pool_key
        self.generation, self.size = generation, size

    def initialize(self):
        with self.registry.connect() as db, db.transaction():
            # PostgreSQL's IF NOT EXISTS does not serialize concurrent DDL.
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtext(current_schema()), hashtext('warm-pool-schema'))"
            )
            db.execute("""
                CREATE TABLE IF NOT EXISTS warm_pool_config (
                    namespace TEXT PRIMARY KEY, generation BIGINT NOT NULL,
                    pool_key TEXT NOT NULL, size INTEGER NOT NULL CHECK(size BETWEEN 0 AND 10),
                    retry_after TIMESTAMPTZ, last_error TEXT
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS warm_pool_slots (
                    namespace TEXT NOT NULL, slot_id TEXT NOT NULL, pool_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('preparing','ready','claimed','retiring','retired')),
                    metadata JSONB, error TEXT, claimed_session_key TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    checked_at TIMESTAMPTZ, expires_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY(namespace, slot_id)
                )
            """)
            # The deployment publisher stamps one generation into app.yaml. Old
            # replicas cannot replenish after a newer rollout lowers/disables it.
            db.execute(
                "INSERT INTO warm_pool_config(namespace,generation,pool_key,size) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT(namespace) DO UPDATE SET "
                "generation=EXCLUDED.generation,pool_key=EXCLUDED.pool_key,size=EXCLUDED.size,"
                "retry_after=NULL,last_error=NULL "
                "WHERE warm_pool_config.generation < EXCLUDED.generation",
                (self.registry.namespace, self.generation, self.key, self.size),
            )

    def current(self, db=None):
        if db is None:
            with self.registry.connect() as connection:
                return self.current(connection)
        row = db.execute(
            "SELECT * FROM warm_pool_config WHERE namespace=%s",
            (self.registry.namespace,),
        ).fetchone()
        return bool(
            row
            and (row["generation"], row["pool_key"], row["size"])
            == (self.generation, self.key, self.size)
        )

    def slots(self):
        with self.registry.connect() as db:
            return db.execute(
                "SELECT *, expires_at <= now() AS expired FROM warm_pool_slots "
                "WHERE namespace=%s AND status NOT IN ('claimed','retired') ORDER BY created_at",
                (self.registry.namespace,),
            ).fetchall()

    def reserve(self):
        """Called under the deployment's controller advisory lock."""
        with self.registry.connect() as db, db.transaction():
            policy = db.execute(
                "SELECT *, COALESCE(retry_after > now(),false) AS cooling "
                "FROM warm_pool_config WHERE namespace=%s FOR UPDATE",
                (self.registry.namespace,),
            ).fetchone()
            if not self.current(db) or policy["cooling"]:
                return None
            count = db.execute(
                "SELECT count(*) AS count FROM warm_pool_slots WHERE namespace=%s "
                "AND status NOT IN ('claimed','retired')",
                (self.registry.namespace,),
            ).fetchone()["count"]
            if count >= self.size:
                return None
            slot_id = "pool:" + uuid4().hex
            return db.execute(
                "INSERT INTO warm_pool_slots(namespace,slot_id,pool_key,status,expires_at) "
                "VALUES (%s,%s,%s,'preparing',now()+interval '1 hour') RETURNING *",
                (self.registry.namespace, slot_id, self.key),
            ).fetchone()

    def mark(self, slot_id, status, *, expected, metadata=None, error=None, checked=False):
        with self.registry.connect() as db:
            return (
                db.execute(
                    "UPDATE warm_pool_slots SET status=%s,metadata=COALESCE(%s,metadata),"
                    "error=%s,updated_at=now(),checked_at=CASE WHEN %s THEN now() ELSE checked_at END "
                    "WHERE namespace=%s AND slot_id=%s AND status=%s RETURNING slot_id",
                    (
                        status,
                        Jsonb(metadata) if metadata is not None else None,
                        error[:2000] if error else None,
                        checked,
                        self.registry.namespace,
                        slot_id,
                        expected,
                    ),
                ).fetchone()
                is not None
            )

    def cooldown(self, error):
        with self.registry.connect() as db:
            db.execute(
                "UPDATE warm_pool_config SET retry_after=now()+interval '60 seconds',last_error=%s "
                "WHERE namespace=%s AND generation=%s",
                (str(error)[:2000], self.registry.namespace, self.generation),
            )

    def recovered(self):
        with self.registry.connect() as db:
            db.execute(
                "UPDATE warm_pool_config SET retry_after=NULL,last_error=NULL "
                "WHERE namespace=%s AND generation=%s",
                (self.registry.namespace, self.generation),
            )

    def finish_retirement(self, slot_id):
        """Forget the placeholder only after shutdown has been confirmed."""
        with self.registry.connect() as db, db.transaction():
            retired = db.execute(
                "UPDATE warm_pool_slots SET status='retired',updated_at=now() "
                "WHERE namespace=%s AND slot_id=%s AND status='retiring' RETURNING slot_id",
                (self.registry.namespace, slot_id),
            ).fetchone()
            if retired:
                db.execute(
                    "DELETE FROM sessions WHERE namespace=%s AND session_key=%s",
                    (self.registry.namespace, self.registry.session_key(slot_id)),
                )

    def status(self):
        with self.registry.connect() as db:
            policy = db.execute(
                "SELECT size,last_error FROM warm_pool_config WHERE namespace=%s",
                (self.registry.namespace,),
            ).fetchone()
            counts = db.execute(
                "SELECT status,count(*) AS count FROM warm_pool_slots WHERE namespace=%s "
                "AND status NOT IN ('claimed','retired') GROUP BY status",
                (self.registry.namespace,),
            ).fetchall()
        return {
            "maximum": policy["size"],
            "ready": 0,
            "preparing": 0,
            "retiring": 0,
            **{r["status"]: r["count"] for r in counts},
            "error": policy["last_error"],
        }

    def claim(self, session, healthy):
        """Atomically move an unused slot's mapping; keep its physical worker directory."""
        db = session.db
        with db.transaction():
            db.execute(
                "SELECT namespace FROM warm_pool_config WHERE namespace=%s FOR SHARE",
                (self.registry.namespace,),
            )
            if not self.size or not self.current(db):
                return False
            slot = db.execute(
                "SELECT * FROM warm_pool_slots WHERE namespace=%s AND pool_key=%s "
                "AND status='ready' AND expires_at > now() ORDER BY created_at "
                "LIMIT 1 FOR UPDATE SKIP LOCKED",
                (self.registry.namespace, self.key),
            ).fetchone()
            if slot is None:
                return False
            # Checking never starts/resumes a worker. A failed probe quarantines
            # the spare; the ordinary session allocation path remains available.
            if not healthy(slot):
                db.execute(
                    "UPDATE warm_pool_slots SET status='retiring',error='Worker health check failed',"
                    "updated_at=now() WHERE namespace=%s AND slot_id=%s",
                    (self.registry.namespace, slot["slot_id"]),
                )
                return False
            from databricks_agentkit.sandbox.store import Session

            key = self.registry.session_key(slot["slot_id"])
            source = Session(db, self.registry.namespace, key).get()
            target = session.get()
            if not source or not target or target["sandbox_id"]:
                raise RuntimeError("Warm pool mapping changed during assignment")
            if source["execution_principal"] != target["execution_principal"]:
                raise RuntimeError("Warm pool execution identity mismatch")
            if (
                source["sandbox_id"] != slot["metadata"]["sandbox_id"]
                or source["status"] != "ready"
            ):
                raise RuntimeError("Warm pool sandbox mapping does not match the prepared worker")
            metrics = {
                **(source["bootstrap_metrics"] or {}),
                "worker_state_key": key,
                "warm_pool": {
                    "slot_id": slot["slot_id"],
                    "prepared_at": slot["updated_at"].isoformat(),
                },
            }
            session.save(
                source["source_hash"],
                source["sandbox_id"],
                "ready",
                metrics,
                source_revision=source["source_revision"],
            )
            db.execute(
                "DELETE FROM sessions WHERE namespace=%s AND session_key=%s",
                (self.registry.namespace, key),
            )
            db.execute(
                "UPDATE warm_pool_slots SET status='claimed',claimed_session_key=%s,updated_at=now() "
                "WHERE namespace=%s AND slot_id=%s",
                (session.key, self.registry.namespace, slot["slot_id"]),
            )
            return True

    def prune(self):
        with self.registry.connect() as db:
            db.execute(
                "DELETE FROM warm_pool_slots WHERE namespace=%s "
                "AND status IN ('claimed','retired') AND updated_at < now()-interval '1 day'",
                (self.registry.namespace,),
            )
