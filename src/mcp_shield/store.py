"""The engine's local SQLite store in the per-user data directory (02 §8, 07 §1).

Pin tables, the event store behind the dashboard (segments redacted, 7-day retention),
live taint records and probe counters (so a restarted engine comes back with the taint
it had), approve-then-retry grants, allowed flows, *Allow this exact content* overrides,
confirmed-attack signatures and the dataset-review queue.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import data_dir, ensure_private_dir

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS servers (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'quarantined',
    transport TEXT NOT NULL DEFAULT 'stdio',
    remote INTEGER NOT NULL DEFAULT 0,
    command TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    status_reason TEXT
);
CREATE TABLE IF NOT EXISTS pins (
    server TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    accepted_hash TEXT,
    accepted_def TEXT,
    current_hash TEXT NOT NULL,
    current_def TEXT NOT NULL,
    privilege TEXT,
    scan_decision TEXT,
    scan_label TEXT,
    scan_prob REAL,
    scanned REAL,
    first_seen REAL NOT NULL,
    updated REAL NOT NULL,
    PRIMARY KEY (server, kind, name)
);
CREATE TABLE IF NOT EXISTS events (
    ref TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    verdict TEXT NOT NULL,
    tier TEXT,
    cls TEXT,
    prob REAL,
    server TEXT,
    tool TEXT,
    direction TEXT,
    method TEXT,
    session TEXT,
    trace_id TEXT,
    reason TEXT,
    segment TEXT,
    segment_sha TEXT,
    origin TEXT NOT NULL DEFAULT 'proxy',
    status TEXT NOT NULL DEFAULT 'open',
    extra TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE TABLE IF NOT EXISTS taint_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    resource TEXT,
    sha256 TEXT NOT NULL,
    fingerprints BLOB NOT NULL,
    embeddings BLOB,
    blobs TEXT,
    ts REAL NOT NULL,
    trace_id TEXT
);
CREATE INDEX IF NOT EXISTS taint_ts ON taint_records (ts);
CREATE TABLE IF NOT EXISTS sessions (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS probes (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS probes_key ON probes (scope, key, ts);
CREATE TABLE IF NOT EXISTS grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    expires REAL NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    ref TEXT
);
CREATE TABLE IF NOT EXISTS flow_grants (
    source TEXT NOT NULL,
    dest TEXT NOT NULL,
    expires REAL NOT NULL,
    ref TEXT,
    PRIMARY KEY (source, dest)
);
CREATE TABLE IF NOT EXISTS allow_content (
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    expires REAL NOT NULL,
    ref TEXT,
    PRIMARY KEY (server, tool, sha256)
);
CREATE TABLE IF NOT EXISTS signatures (
    sha256 TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB,
    added REAL NOT NULL,
    ref TEXT
);
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    label TEXT,
    context TEXT,
    segment TEXT NOT NULL,
    ref TEXT
);
CREATE TABLE IF NOT EXISTS tool_caps (
    server TEXT NOT NULL,
    tool TEXT NOT NULL,
    cap INTEGER NOT NULL,
    PRIMARY KEY (server, tool)
);
"""


@dataclass
class PinRow:
    server: str
    kind: str
    name: str
    accepted_hash: str | None
    accepted_def: dict[str, Any] | None
    current_hash: str
    current_def: dict[str, Any]
    privilege: str | None
    scan_decision: str | None
    scan_label: str | None
    scan_prob: float | None
    scanned: float | None
    first_seen: float
    updated: float

    @property
    def changed(self) -> bool:
        return self.accepted_hash is not None and self.accepted_hash != self.current_hash

    @property
    def accepted(self) -> bool:
        return self.accepted_hash is not None and self.accepted_hash == self.current_hash


class Store:
    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = ensure_private_dir(data_dir()) / "mcp-shield.db"
        else:
            ensure_private_dir(path.parent)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _q(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, args).fetchall())

    def _x(self, sql: str, args: tuple[Any, ...] = ()) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ------------------------------------------------------------------ servers

    def touch_server(self, name: str, transport: str, remote: bool, command: str | None = None) -> dict[str, Any]:
        now = time.time()
        with self.tx() as db:
            db.execute(
                "INSERT INTO servers (name, transport, remote, command, first_seen, last_seen) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET transport=excluded.transport, remote=excluded.remote, "
                "command=COALESCE(excluded.command, servers.command), last_seen=excluded.last_seen",
                (name, transport, int(remote), command, now, now),
            )
        server = self.server(name)
        assert server is not None
        return server

    def server(self, name: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM servers WHERE name=?", (name,))
        return dict(rows[0]) if rows else None

    def servers(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM servers ORDER BY name")]

    def set_server_status(self, name: str, status: str, reason: str | None = None) -> None:
        self._x("UPDATE servers SET status=?, status_reason=? WHERE name=?", (status, reason, name))

    # ------------------------------------------------------------------ pins

    def pins(self, server: str) -> dict[tuple[str, str], PinRow]:
        out: dict[tuple[str, str], PinRow] = {}
        for r in self._q("SELECT * FROM pins WHERE server=?", (server,)):
            out[(r["kind"], r["name"])] = PinRow(
                r["server"],
                r["kind"],
                r["name"],
                r["accepted_hash"],
                json.loads(r["accepted_def"]) if r["accepted_def"] else None,
                r["current_hash"],
                json.loads(r["current_def"]),
                r["privilege"],
                r["scan_decision"],
                r["scan_label"],
                r["scan_prob"],
                r["scanned"],
                r["first_seen"],
                r["updated"],
            )
        return out

    def upsert_pin(
        self,
        server: str,
        kind: str,
        name: str,
        current_hash: str,
        current_def: dict[str, Any],
        privilege: str | None,
        scan: tuple[str, str, float, float] | None,
        *,
        accept: bool = False,
    ) -> None:
        """Record the current definition; ``scan`` is ``(decision, label, probability, scanned_at)``."""
        now = time.time()
        decision, label, prob, scanned = scan if scan else (None, None, None, None)
        def_json = json.dumps(current_def, sort_keys=True, ensure_ascii=False)
        with self.tx() as db:
            db.execute(
                "INSERT INTO pins (server, kind, name, accepted_hash, accepted_def, current_hash, current_def,"  # noqa: S608
                " privilege, scan_decision, scan_label, scan_prob, scanned, first_seen, updated)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(server, kind, name) DO UPDATE SET current_hash=excluded.current_hash, "
                "current_def=excluded.current_def, privilege=excluded.privilege, scan_decision=excluded.scan_decision, "
                "scan_label=excluded.scan_label, scan_prob=excluded.scan_prob, scanned=excluded.scanned, "
                "updated=excluded.updated"
                + (", accepted_hash=excluded.accepted_hash, accepted_def=excluded.accepted_def" if accept else ""),
                (
                    server,
                    kind,
                    name,
                    current_hash if accept else None,
                    def_json if accept else None,
                    current_hash,
                    def_json,
                    privilege,
                    decision,
                    label,
                    prob,
                    scanned,
                    now,
                    now,
                ),
            )

    def accept_pins(self, server: str, items: list[tuple[str, str]] | None = None) -> int:
        """Accept the current definitions (all, or the listed ``(kind, name)`` items). Returns the count."""
        with self.tx() as db:
            if items is None:
                n = db.execute(
                    "UPDATE pins SET accepted_hash=current_hash, accepted_def=current_def WHERE server=?",
                    (server,),
                ).rowcount
            else:
                n = 0
                for kind, name in items:
                    n += db.execute(
                        "UPDATE pins SET accepted_hash=current_hash, accepted_def=current_def "
                        "WHERE server=? AND kind=? AND name=?",
                        (server, kind, name),
                    ).rowcount
            db.execute("UPDATE servers SET status='accepted', status_reason=NULL WHERE name=?", (server,))
        return n

    def forget_server(self, server: str) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM pins WHERE server=?", (server,))
            db.execute("DELETE FROM servers WHERE name=?", (server,))

    # ------------------------------------------------------------------ events

    def ref_exists(self, ref: str) -> bool:
        return bool(self._q("SELECT 1 FROM events WHERE ref=?", (ref,)))

    def add_event(self, event: dict[str, Any]) -> None:
        cols = (
            "ref",
            "ts",
            "kind",
            "verdict",
            "tier",
            "cls",
            "prob",
            "server",
            "tool",
            "direction",
            "method",
            "session",
            "trace_id",
            "reason",
            "segment",
            "segment_sha",
            "origin",
            "status",
        )
        values = [event.get(c) for c in cols]
        extra = {k: v for k, v in event.items() if k not in cols}
        if values[cols.index("origin")] is None:
            values[cols.index("origin")] = "proxy"
        if values[cols.index("status")] is None:
            values[cols.index("status")] = "open"
        self._x(
            # column names are the constants above; every value is a bound parameter
            f"INSERT INTO events ({', '.join(cols)}, extra) VALUES ({', '.join('?' * len(cols))}, ?)",  # noqa: S608
            (*values, json.dumps(extra, default=str) if extra else None),
        )

    @staticmethod
    def _event_row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        extra = d.pop("extra", None)
        if extra:
            d.update(json.loads(extra))
        return d

    def event(self, ref: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM events WHERE ref=?", (ref,))
        return self._event_row(rows[0]) if rows else None

    def events(
        self,
        limit: int = 200,
        *,
        verdicts: list[str] | None = None,
        server: str | None = None,
        since: float | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        args: list[Any] = []
        if verdicts:
            where.append(f"verdict IN ({', '.join('?' * len(verdicts))})")
            args.extend(verdicts)
        if server:
            where.append("server=?")
            args.append(server)
        if since:
            where.append("ts>=?")
            args.append(since)
        if query:
            where.append("(ref LIKE ? OR tool LIKE ? OR server LIKE ? OR reason LIKE ?)")
            args.extend([f"%{query}%"] * 4)
        clause = " WHERE " + " AND ".join(where) if where else ""  # fixed fragments; values are bound parameters
        sql = "SELECT * FROM events" + clause + " ORDER BY ts DESC LIMIT ?"  # noqa: S608
        return [self._event_row(r) for r in self._q(sql, (*args, limit))]

    def set_event_status(self, ref: str, status: str) -> None:
        self._x("UPDATE events SET status=? WHERE ref=?", (status, ref))

    # ------------------------------------------------------------------ taint + sessions

    def add_taint_record(self, rec: dict[str, Any]) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO taint_records (session, server, tool, resource, sha256, fingerprints, embeddings, blobs,"
                " ts, trace_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["session"],
                    rec["server"],
                    rec["tool"],
                    rec.get("resource"),
                    rec["sha256"],
                    rec["fingerprints"],
                    rec.get("embeddings"),
                    rec.get("blobs"),
                    rec["ts"],
                    rec.get("trace_id"),
                ),
            )
            return int(cur.lastrowid or 0)

    def refresh_taint_record(self, record_id: int, ts: float) -> None:
        self._x("UPDATE taint_records SET ts=? WHERE id=?", (ts, record_id))

    def taint_records(self, since: float) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM taint_records WHERE ts>=? ORDER BY ts", (since,))]

    def delete_taint_records(self, session: str | None = None, before: float | None = None) -> None:
        if session is not None:
            self._x("DELETE FROM taint_records WHERE session=?", (session,))
        if before is not None:
            self._x("DELETE FROM taint_records WHERE ts<?", (before,))

    def save_session(self, key: str, data: dict[str, Any], last_seen: float) -> None:
        self._x(
            "INSERT INTO sessions (key, data, last_seen) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET data=excluded.data, last_seen=excluded.last_seen",
            (key, json.dumps(data), last_seen),
        )

    def sessions(self, since: float) -> list[tuple[str, dict[str, Any], float]]:
        return [
            (r["key"], json.loads(r["data"]), r["last_seen"])
            for r in self._q("SELECT * FROM sessions WHERE last_seen>=?", (since,))
        ]

    def add_probe(self, scope: str, key: str, ts: float) -> None:
        self._x("INSERT INTO probes (scope, key, ts) VALUES (?,?,?)", (scope, key, ts))

    def count_probes(self, scope: str, key: str, since: float) -> int:
        return int(
            self._q("SELECT COUNT(*) AS n FROM probes WHERE scope=? AND key=? AND ts>=?", (scope, key, since))[0]["n"]
        )

    # ------------------------------------------------------------------ grants and overrides

    def add_grant(self, session: str, server: str, tool: str, args_hash: str, expires: float, ref: str | None) -> None:
        self._x(
            "INSERT INTO grants (session, server, tool, args_hash, expires, ref) VALUES (?,?,?,?,?,?)",
            (session, server, tool, args_hash, expires, ref),
        )

    def consume_grant(self, session: str, server: str, tool: str, args_hash: str, now: float) -> bool:
        with self.tx() as db:
            row = db.execute(
                "SELECT id FROM grants WHERE session=? AND server=? AND tool=? AND args_hash=? AND used=0"
                " AND expires>=? ORDER BY id LIMIT 1",
                (session, server, tool, args_hash, now),
            ).fetchone()
            if row is None:
                return False
            db.execute("UPDATE grants SET used=1 WHERE id=?", (row["id"],))
            return True

    def add_flow_grant(self, source: str, dest: str, expires: float, ref: str | None) -> None:
        self._x(
            "INSERT INTO flow_grants (source, dest, expires, ref) VALUES (?,?,?,?) "
            "ON CONFLICT(source, dest) DO UPDATE SET expires=excluded.expires, ref=excluded.ref",
            (source, dest, expires, ref),
        )

    def flow_granted(self, source: str, dest: str, now: float) -> bool:
        return bool(self._q("SELECT 1 FROM flow_grants WHERE source=? AND dest=? AND expires>=?", (source, dest, now)))

    def add_allow_content(self, server: str, tool: str, sha256: str, expires: float, ref: str | None) -> None:
        self._x(
            "INSERT INTO allow_content (server, tool, sha256, expires, ref) VALUES (?,?,?,?,?) "
            "ON CONFLICT(server, tool, sha256) DO UPDATE SET expires=excluded.expires, ref=excluded.ref",
            (server, tool, sha256, expires, ref),
        )

    def allow_contents(self, now: float) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM allow_content WHERE expires>=?", (now,))]

    def add_signature(self, sha256: str, text: str, embedding: bytes | None, ref: str | None) -> None:
        self._x(
            "INSERT OR REPLACE INTO signatures (sha256, text, embedding, added, ref) VALUES (?,?,?,?,?)",
            (sha256, text, embedding, time.time(), ref),
        )

    def signatures(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM signatures")]

    def add_review(self, kind: str, segment: str, label: str | None, context: str | None, ref: str | None) -> None:
        self._x(
            "INSERT INTO review_queue (ts, kind, label, context, segment, ref) VALUES (?,?,?,?,?,?)",
            (time.time(), kind, label, context, segment, ref),
        )

    def review_queue(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM review_queue ORDER BY ts")]

    def set_tool_cap(self, server: str, tool: str, cap: int) -> None:
        self._x(
            "INSERT INTO tool_caps (server, tool, cap) VALUES (?,?,?) "
            "ON CONFLICT(server, tool) DO UPDATE SET cap=excluded.cap",
            (server, tool, cap),
        )

    def tool_cap(self, server: str, tool: str) -> int | None:
        rows = self._q("SELECT cap FROM tool_caps WHERE server=? AND tool=?", (server, tool))
        return int(rows[0]["cap"]) if rows else None

    # ------------------------------------------------------------------ retention

    def purge(self, now: float, retention_days: int, taint_horizon_s: float) -> None:
        cutoff = now - retention_days * 86400
        with self.tx() as db:
            db.execute("UPDATE events SET segment=NULL WHERE ts<? AND segment IS NOT NULL", (cutoff,))
            db.execute("DELETE FROM events WHERE ts<?", (cutoff - 23 * 86400,))
            db.execute("DELETE FROM taint_records WHERE ts<?", (now - taint_horizon_s,))
            db.execute("DELETE FROM probes WHERE ts<?", (now - 86400,))
            db.execute("DELETE FROM grants WHERE expires<?", (now - 86400,))
            db.execute("DELETE FROM flow_grants WHERE expires<?", (now,))
            db.execute("DELETE FROM allow_content WHERE expires<?", (now,))
            db.execute("DELETE FROM sessions WHERE last_seen<?", (now - 86400,))
