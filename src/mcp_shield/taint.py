"""Session taint state (02 §4) — held in the engine, outside the LLM.

* **Session key** — bearer subject + ``clientInfo`` + host identity, never the server, so
  one host talking to five servers is one session and taint follows data between them.
  Trace context is recorded, never used to partition.
* **Records** — every untrusted result forwarded to the host becomes a
  ``TaintRecord`` with winnowed fingerprints, embeddings and blob hashes. A record's
  weight halves every 15 minutes and stops counting below ¼ (30 minutes); re-reading
  the same content refreshes it.
* **Containment check** — (a) every ``write`` / ``execute`` / ``egress`` call and (b) every
  call to a *remote* server other than the record's source. Write-back to the same
  server *and* resource is exempt; trusted flows are exempt and logged.
* **Lookup scope** — the containment check reads the live records of every session that
  shares the bearer subject and client name ("taint group"), so the same host reaching a
  stdio front-end and an HTTP listener (which see different host identities) is still
  one flow. Over-tainting is the safe direction.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .cascade.privilege import EGRESS, EXECUTE, HIGH, WRITE
from .config import SecurityConfig
from .store import Store
from .winnow import Fingerprints, fingerprint, longest_shared_run

_HOW_ORDER = {"shared-run": 0, "fingerprint-hit": 1, "blob": 2, "embedding": 3}
SEP = "\x1f"  # joins a result's segments; never appears in a canonical argument
MAX_RECORDS_PER_GROUP = 512
MAX_TEXT_PER_GROUP = 8 * 1024 * 1024
MAX_EMBEDDINGS_PER_RECORD = 16
MIN_ARG_CHARS = 12


def short_hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def args_hash(server: str, tool: str, arguments: Any) -> str:
    """The exact-call key an *Allow once* grant is bound to."""
    canonical = json.dumps([server, tool, arguments], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Session:
    key: str
    group: str
    subject: str
    client: str
    host: str
    created: float
    last_seen: float
    policy: str = "standard"
    strict_reasons: list[str] = field(default_factory=list)
    quarantined: bool = False
    trace_ids: list[str] = field(default_factory=list)

    @property
    def strict(self) -> bool:
        return self.policy == "strict"

    def escalate(self, reason: str) -> bool:
        changed = self.policy != "strict"
        self.policy = "strict"
        if reason not in self.strict_reasons:
            self.strict_reasons.append(reason)
        return changed

    def to_json(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "subject": self.subject,
            "client": self.client,
            "host": self.host,
            "created": self.created,
            "policy": self.policy,
            "strict_reasons": self.strict_reasons,
            "quarantined": self.quarantined,
            "trace_ids": self.trace_ids[-5:],
        }


@dataclass
class TaintRecord:
    id: int
    session: str
    group: str
    server: str
    tool: str
    resource: str | None
    sha256: str
    fingerprints: Fingerprints
    canonical: str | None
    embeddings: NDArray[Any] | None
    blobs: frozenset[str]
    ts: float
    trace_id: str | None = None

    def weight(self, now: float, half_life: float) -> float:
        return float(0.5 ** (max(0.0, now - self.ts) / half_life))


@dataclass
class Overlap:
    record: TaintRecord
    how: str  # "shared-run" | "embedding" | "blob"
    detail: str
    argument: str


@dataclass
class TaintDecision:
    checked: bool = False
    overlaps: list[Overlap] = field(default_factory=list)
    allowed_flows: list[Overlap] = field(default_factory=list)
    exempt_write_back: int = 0

    @property
    def violation(self) -> bool:
        return bool(self.overlaps)

    @property
    def sources(self) -> list[str]:
        return sorted({o.record.server for o in self.overlaps})


class TaintEngine:
    def __init__(self, config: SecurityConfig, store: Store | None = None, clock: Any = time.time) -> None:
        self.config = config
        self.store = store
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        self.records: OrderedDict[int, TaintRecord] = OrderedDict()
        self._next_local_id = -1
        self.live_horizon = config.taint_half_life_s * float(np.log2(1 / config.taint_live_weight))
        if store is not None:
            self._restore()

    # ------------------------------------------------------------------ sessions

    def session_for(
        self, subject: str | None, client_name: str, client_version: str, host: str, trace_id: str | None = None
    ) -> Session:
        now = self.clock()
        subject = subject or ""
        key = short_hash("session", subject, client_name, client_version, host)
        session = self.sessions.get(key)
        if session is not None and now - session.last_seen > self.config.idle_window_s:
            session = None  # aged out: a new session starts clean
        if session is None:
            session = Session(
                key,
                short_hash("group", subject, client_name),
                subject,
                f"{client_name}/{client_version}".strip("/"),
                host,
                now,
                now,
            )
            if self.config.taint_policy == "strict":
                session.escalate("config")
            self.sessions[key] = session
        session.last_seen = now
        if trace_id and trace_id not in session.trace_ids:
            session.trace_ids.append(trace_id)
            del session.trace_ids[:-20]
        self._save_session(session)
        return session

    def _save_session(self, session: Session) -> None:
        if self.store is not None:
            self.store.save_session(session.key, session.to_json(), session.last_seen)

    def escalate(self, session: Session, reason: str) -> bool:
        changed = session.escalate(reason)
        self._save_session(session)
        return changed

    def reset(self, session: Session) -> None:
        session.policy = "strict" if self.config.taint_policy == "strict" else "standard"
        session.strict_reasons = ["config"] if session.policy == "strict" else []
        session.quarantined = False
        self._save_session(session)

    # ------------------------------------------------------------------ records

    def live(self, group: str, now: float | None = None) -> list[TaintRecord]:
        now = self.clock() if now is None else now
        floor = self.config.taint_live_weight
        return [
            r
            for r in self.records.values()
            if r.group == group and r.weight(now, self.config.taint_half_life_s) >= floor
        ]

    def tainted(self, session: Session) -> bool:
        return bool(self.live(session.group))

    def record(
        self,
        session: Session,
        server: str,
        tool: str,
        resource: str | None,
        texts: list[str],
        blobs: list[str],
        embeddings: list[NDArray[Any]],
        trace_id: str | None = None,
    ) -> TaintRecord | None:
        if not texts and not blobs:
            return None
        now = self.clock()
        canonical, fps = fingerprint(SEP.join(texts))
        sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        for rec in self.records.values():
            if rec.group == session.group and rec.sha256 == sha and rec.server == server:
                rec.ts = now  # reading the same content again refreshes it
                rec.resource = resource or rec.resource
                self.records.move_to_end(rec.id)
                if self.store is not None and rec.id > 0:
                    self.store.refresh_taint_record(rec.id, now)
                return rec
        emb = None
        if embeddings:
            stack = np.stack(
                [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings[:MAX_EMBEDDINGS_PER_RECORD]]
            )
            norms = np.linalg.norm(stack, axis=1, keepdims=True)
            emb = (stack / np.where(norms == 0, 1, norms)).astype(np.float16)
        blob_set = frozenset(blobs)
        rec_id = self._persist(session, server, tool, resource, sha, fps, emb, blob_set, now, trace_id)
        rec = TaintRecord(
            rec_id,
            session.key,
            session.group,
            server,
            tool,
            resource,
            sha,
            fps,
            canonical,
            emb,
            blob_set,
            now,
            trace_id,
        )
        self.records[rec_id] = rec
        self._bound(session.group)
        return rec

    def _persist(
        self,
        session: Session,
        server: str,
        tool: str,
        resource: str | None,
        sha: str,
        fps: Fingerprints,
        emb: NDArray[Any] | None,
        blobs: frozenset[str],
        now: float,
        trace_id: str | None,
    ) -> int:
        if self.store is None:
            self._next_local_id -= 1
            return self._next_local_id
        return self.store.add_taint_record(
            {
                "session": f"{session.key}:{session.group}",
                "server": server,
                "tool": tool,
                "resource": resource,
                "sha256": sha,
                "fingerprints": fps.to_bytes(),
                "embeddings": None if emb is None else _emb_bytes(emb),
                "blobs": json.dumps(sorted(blobs)) if blobs else None,
                "ts": now,
                "trace_id": trace_id,
            }
        )

    def _bound(self, group: str) -> None:
        mine = [r for r in self.records.values() if r.group == group]
        for rec in mine[: max(0, len(mine) - MAX_RECORDS_PER_GROUP)]:
            del self.records[rec.id]
        text = 0
        for rec in reversed([r for r in self.records.values() if r.group == group]):
            if rec.canonical is not None:
                text += len(rec.canonical)
                if text > MAX_TEXT_PER_GROUP:
                    rec.canonical = None  # falls back to the conservative any-hit rule

    def _restore(self) -> None:
        assert self.store is not None
        now = self.clock()
        for key, data, last_seen in self.store.sessions(now - self.config.idle_window_s):
            self.sessions[key] = Session(
                key,
                data["group"],
                data.get("subject", ""),
                data.get("client", ""),
                data.get("host", ""),
                data.get("created", last_seen),
                last_seen,
                data.get("policy", "standard"),
                list(data.get("strict_reasons", [])),
                bool(data.get("quarantined", False)),
                list(data.get("trace_ids", [])),
            )
        for row in self.store.taint_records(now - self.live_horizon):
            session_key, _, group = str(row["session"]).partition(":")
            emb = _emb_from_bytes(row["embeddings"]) if row["embeddings"] else None
            blobs = frozenset(json.loads(row["blobs"])) if row.get("blobs") else frozenset()
            self.records[row["id"]] = TaintRecord(
                row["id"],
                session_key,
                group or session_key,
                row["server"],
                row["tool"],
                row["resource"],
                row["sha256"],
                Fingerprints.from_bytes(row["fingerprints"]),
                None,
                emb,
                blobs,
                row["ts"],
                row["trace_id"],
            )

    def expire(self) -> None:
        now = self.clock()
        floor = self.config.taint_live_weight
        for rid in [rid for rid, r in self.records.items() if r.weight(now, self.config.taint_half_life_s) < floor]:
            del self.records[rid]
        for key in [k for k, s in self.sessions.items() if now - s.last_seen > self.config.idle_window_s]:
            del self.sessions[key]

    # ------------------------------------------------------------------ the containment check

    def check_call(
        self,
        session: Session,
        server: str,
        privilege: str,
        remote: bool,
        arg_strings: list[tuple[str, str]],
        arg_embeddings: list[tuple[str, NDArray[Any]]] | None = None,
        resource: str | None = None,
        flow_allowed: Any = None,
        cosine_threshold: float | None = None,
    ) -> TaintDecision:
        """Does this call carry live tainted content somewhere it is checked?

        ``cosine_threshold`` is the model's calibrated near-paraphrase threshold; the config
        value acts as a floor. Without embeddings the check is the exact shared run alone.
        """
        rule_a = privilege in HIGH
        rule_b = remote
        decision = TaintDecision(checked=rule_a or rule_b)
        if not decision.checked:
            return decision
        candidates: list[TaintRecord] = []
        for rec in self.live(session.group):
            if rule_a:
                if privilege == WRITE and rec.server == server and resource and rec.resource == resource:
                    decision.exempt_write_back += 1
                    continue
                candidates.append(rec)
            elif rec.server != server:
                candidates.append(rec)
        if not candidates:
            return decision

        queries = [(path, fingerprint(text)[0], text) for path, text in arg_strings if len(text) >= MIN_ARG_CHARS]
        arg_shas = {hashlib.sha256(text.encode("utf-8")).hexdigest(): path for path, text in arg_strings}
        for rec in candidates:
            threshold = max(self.config.cosine_threshold, cosine_threshold or 0.0)
            overlap = self._overlap(rec, queries, arg_embeddings or [], arg_shas, threshold)
            if overlap is None:
                continue
            allowed = (flow_allowed(rec.server, server) if flow_allowed else False) or self.config.flow_allowed(
                rec.server, server
            )
            (decision.allowed_flows if allowed and rec.server != server else decision.overlaps).append(overlap)
        decision.overlaps.sort(key=lambda o: _HOW_ORDER.get(o.how, 9))
        return decision

    def _overlap(
        self,
        rec: TaintRecord,
        queries: list[tuple[str, str, str]],
        arg_embeddings: list[tuple[str, NDArray[Any]]],
        arg_shas: dict[str, str],
        threshold: float,
    ) -> Overlap | None:
        min_run = self.config.overlap_min_run
        for path, canonical, _raw in queries:
            run = longest_shared_run(canonical, rec.canonical, rec.fingerprints, min_run)
            if run >= min_run:
                how = "shared-run" if rec.canonical is not None else "fingerprint-hit"
                return Overlap(rec, how, f"{run}-character run shared with {rec.server}/{rec.tool}", path)
        if rec.embeddings is not None and arg_embeddings:
            recs = rec.embeddings.astype(np.float32)
            for path, emb in arg_embeddings:
                v = np.asarray(emb, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(v))
                if n == 0 or v.shape[0] != recs.shape[1]:
                    continue
                cos = float(np.max(recs @ (v / n)))
                if cos >= threshold:
                    return Overlap(rec, "embedding", f"cosine {cos:.2f} with {rec.server}/{rec.tool}", path)
        for sha, path in arg_shas.items():
            if sha in rec.blobs:
                return Overlap(rec, "blob", f"binary content from {rec.server}/{rec.tool}", path)
        return None

    # ------------------------------------------------------------------ strict policy

    def needs_hold(self, session: Session, privilege: str, *, shell_mode: bool, annotated: bool) -> str | None:
        """Why an ``execute`` / ``egress`` call must be approved first, or ``None``."""
        if privilege not in (EXECUTE, EGRESS):
            return None
        if not self.tainted(session):
            return None
        if shell_mode:
            return "shell-mode tool in a tainted session"
        if session.strict:
            return "strict policy: " + ", ".join(session.strict_reasons or ["config"])
        if annotated:
            return "argument policy annotation on an execute/egress call in a tainted session"
        return None


def _emb_bytes(emb: NDArray[Any]) -> bytes:
    rows, dim = emb.shape
    return np.array([rows, dim], dtype="<u4").tobytes() + emb.astype("<f2").tobytes()


def _emb_from_bytes(data: bytes) -> NDArray[Any]:
    rows, dim = np.frombuffer(data[:8], dtype="<u4")
    return np.frombuffer(data[8:], dtype="<f2").reshape(int(rows), int(dim)).astype(np.float16)
