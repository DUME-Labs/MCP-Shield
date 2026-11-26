"""The engine's per-message decision logic (02 §1–§5).

``Inspector.from_host`` and ``Inspector.from_server`` take one JSON-RPC message from a
front-end or an HTTP listener and return a :class:`Decision`: the messages to send on to
the server and back to the host. A blocked host → server call is never forwarded; a
blocked server → host result is consumed and answered to the host with the opaque stub;
a blocked legacy sampling / elicitation request is answered to the server with the
ordinary decline any host sends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from .. import protocol as P
from .. import spotlight
from ..cascade.argpolicy import FIELD_PATH, FIELD_URL, check_url, field_type_for
from ..cascade.extractor import (
    Segment,
    extract_arguments,
    extract_definition,
    extract_error,
    extract_input_request,
    extract_notification,
    extract_result,
    get_at,
    set_at,
    string_leaves,
)
from ..cascade.privilege import READ, privilege_of
from ..cascade.tier1_policy import check_tool_call, header_body_mismatches
from ..cascade.tier2_cache import ClearedCache, SignatureSet, dos_check
from ..cascade.tier3_classifier import (
    VERDICT_ABSTAIN,
    VERDICT_ALLOW,
    VERDICT_BLOCK,
    Classifier,
    ClassifierUnavailable,
    SegmentInput,
    Verdict,
)
from ..config import SecurityConfig, ToolPolicy
from ..errors import is_stub_message, stub_error
from ..jsonrpc import (
    ERROR,
    NOTIFICATION,
    REQUEST,
    RESPONSE,
    error_response,
    kind_of,
    make_error,
    params_of,
    shape_problem,
)
from ..pinning import (
    LIST_KEYS,
    SCAN_ERROR,
    STATUS_ACCEPTED,
    STATUS_PROBE_QUARANTINED,
    STATUS_QUARANTINED,
    canonical_def,
    def_hash,
    diff_definitions,
    exposure,
    item_name,
    load_known_manifests,
    matches_known_manifest,
)
from ..store import Store
from ..taint import Session, TaintEngine, args_hash
from .events import EventBus, EventLog

log = logging.getLogger(__name__)

SEVERITY = {VERDICT_ALLOW: 0, VERDICT_ABSTAIN: 1, SCAN_ERROR: 2, VERDICT_BLOCK: 3}
RESULT_METHODS = ("tools/call", "resources/read", "prompts/get")
SOURCE_KIND = {"tools/call": "tool", "resources/read": "resource", "prompts/get": "prompt"}


# --------------------------------------------------------------------------- data


@dataclass
class Pending:
    method: str
    tool: str
    resource: str | None
    session_key: str
    started: float
    privilege: str = READ


@dataclass
class Connection:
    """One front-end (stdio) or one HTTP exchange's view of a server."""

    server: str
    transport: str  # "stdio" | "http"
    remote: bool
    host_id: str
    rules: str
    local: bool
    subject: str | None = None
    client_info: dict[str, Any] | None = None
    protocol_version: str | None = None
    session_key: str | None = None
    pending_h2s: dict[str, Pending] = field(default_factory=dict)
    pending_s2h: dict[str, str] = field(default_factory=dict)


@dataclass
class Decision:
    to_server: list[dict[str, Any]] = field(default_factory=list)
    to_host: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "allow"
    ref: str | None = None
    http_status: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "to_server": self.to_server,
            "to_host": self.to_host,
            "verdict": self.verdict,
            "ref": self.ref,
            "http_status": self.http_status,
        }


@dataclass
class SegDecision:
    decision: str
    tier: str
    label: str
    prob: float | None
    reason: str
    verdict: Verdict | None = None


@dataclass
class ScanOutcome:
    decisions: list[SegDecision]
    segments: list[Segment]
    timings: dict[str, float]

    def worst(self, level: str) -> tuple[Segment, SegDecision] | None:
        found = [(s, d) for s, d in zip(self.segments, self.decisions, strict=True) if d.decision == level]
        if not found:
            return None
        return max(found, key=lambda sd: sd[1].prob or 0.0)

    @property
    def blocked(self) -> tuple[Segment, SegDecision] | None:
        return self.worst(VERDICT_BLOCK) or self.worst(SCAN_ERROR)

    @property
    def abstained(self) -> tuple[Segment, SegDecision] | None:
        return self.worst(VERDICT_ABSTAIN)

    def embeddings(self) -> list[tuple[Segment, NDArray[Any]]]:
        out = []
        for s, d in zip(self.segments, self.decisions, strict=True):
            if d.verdict is not None and d.verdict.embedding is not None:
                out.append((s, d.verdict.embedding))
        return out


def _id_key(msg_id: Any) -> str:
    return json.dumps(msg_id)


def _decline(method: str, msg_id: Any) -> dict[str, Any]:
    if method == "elicitation/create":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"action": "decline"}}
    return error_response(msg_id, make_error(P.SAMPLING_USER_REJECTED, "User rejected sampling request"))


def fail_closed(direction: str, message: Any, ref: str) -> Decision:
    """What a front-end does with a message when no verdict can be had (engine down or erroring).

    Every check lives in the engine, so traffic is blocked in *both* directions: a host
    request is answered with the stub, a server response becomes the stub for the host,
    a server request is declined, notifications are dropped.
    """
    kind = kind_of(message) if isinstance(message, dict) else None
    msg_id = message.get("id") if isinstance(message, dict) else None
    if direction == "h2s":
        if kind == REQUEST:
            return Decision(to_host=[error_response(msg_id, stub_error(ref))], verdict="block", ref=ref)
        return Decision(verdict="block", ref=ref)
    if kind in (RESPONSE, ERROR):
        return Decision(to_host=[error_response(msg_id, stub_error(ref))], verdict="block", ref=ref)
    if kind == REQUEST:
        method = str(message.get("method"))
        if method in ("sampling/createMessage", "elicitation/create"):
            return Decision(to_server=[_decline(method, msg_id)], verdict="block", ref=ref)
        return Decision(
            to_server=[error_response(msg_id, make_error(P.METHOD_NOT_FOUND, "Method not found"))],
            verdict="block",
            ref=ref,
        )
    return Decision(verdict="block", ref=ref)


# --------------------------------------------------------------------------- the inspector


class Inspector:
    def __init__(
        self,
        config: SecurityConfig,
        store: Store,
        classifier: Classifier,
        *,
        bus: EventBus | None = None,
        clock: Any = time.time,
    ) -> None:
        self.config = config
        self.store = store
        self.classifier = classifier
        self.clock = clock
        self.log = EventLog(store, bus, clock)
        self.bus = self.log.bus
        self.cache = ClearedCache()
        self.signatures = SignatureSet()
        self.taint = TaintEngine(config, store, clock)
        self.known_manifests = load_known_manifests()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-shield-tier3")
        self._warned: set[tuple[str, str]] = set()
        self.stats: dict[str, int] = {"messages": 0, "blocked": 0, "held": 0}
        self._load_overrides()

    def _load_overrides(self) -> None:
        now = self.clock()
        for row in self.store.allow_contents(now):
            self.cache.put(row["server"], row["tool"], row["sha256"], row["expires"] - now, now)
        for row in self.store.signatures():
            emb = np.frombuffer(row["embedding"], dtype="<f4") if row["embedding"] else None
            self.signatures.add(row["text"], emb)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ connections & sessions

    def register(self, conn: Connection, command: str | None = None) -> None:
        policy = self.config.servers.get(conn.server)
        if policy is not None and policy.remote is not None:
            conn.remote = policy.remote
        self.store.touch_server(conn.server, conn.transport, conn.remote, command)

    def server_status(self, server: str) -> str:
        row = self.store.server(server)
        return str(row["status"]) if row else STATUS_QUARANTINED

    def _session(self, conn: Connection, message: dict[str, Any] | None) -> Session:
        meta = params_of(message).get("_meta") if message else None
        meta = meta if isinstance(meta, dict) else {}
        info = meta.get(P.META_CLIENT_INFO)
        if not isinstance(info, dict):
            info = conn.client_info if isinstance(conn.client_info, dict) else {}
        trace = meta.get("traceparent")
        trace_id = trace.split("-")[1] if isinstance(trace, str) and trace.count("-") >= 3 else None
        session = self.taint.session_for(
            conn.subject, str(info.get("name", "")), str(info.get("version", "")), conn.host_id, trace_id
        )
        conn.session_key = session.key
        return session

    def _conn_session(self, conn: Connection) -> Session:
        if conn.session_key and conn.session_key in self.taint.sessions:
            return self.taint.sessions[conn.session_key]
        return self._session(conn, None)

    # ------------------------------------------------------------------ Tier 2 + Tier 3

    async def _classify(self, inputs: list[SegmentInput]) -> list[Verdict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.classifier.classify, inputs)

    async def scan(self, segments: list[Segment], server: str, tool_key: str, purpose: str) -> ScanOutcome:
        """Tier 2 then Tier 3 over ``segments``; one decision per segment."""
        t0 = time.perf_counter()
        decisions: list[SegDecision | None] = [None] * len(segments)
        todo: list[int] = []
        seen: dict[str, int] = {}
        for i, seg in enumerate(segments):
            why = self.signatures.match(seg.text, seg.sha256)
            if why:
                decisions[i] = SegDecision(VERDICT_BLOCK, "tier2", "signature", 1.0, why)
            elif self.cache.hit(server, tool_key, seg.sha256, self.clock()):
                decisions[i] = SegDecision(VERDICT_ALLOW, "tier2", "benign", None, "cleared content (cache)")
            elif seg.sha256 in seen:
                continue  # same text twice in one message: classify once
            else:
                seen[seg.sha256] = i
                todo.append(i)
        t1 = time.perf_counter()
        if todo:
            inputs = [SegmentInput(segments[i].prefix, segments[i].text, purpose) for i in todo]
            try:
                verdicts = await self._classify(inputs)
            except ClassifierUnavailable as exc:
                for i in todo:
                    if self.config.fail_open:
                        decisions[i] = SegDecision(VERDICT_ALLOW, "tier3", "benign", None, f"FAIL-OPEN: {exc}")
                    else:
                        decisions[i] = SegDecision(
                            SCAN_ERROR, "tier3", "unavailable", None, f"classifier unavailable, failing closed: {exc}"
                        )
                if self.config.fail_open:
                    self._warn_once(("fail-open", str(exc)[:80]), f"--fail-open: content passed unclassified ({exc})")
            else:
                for i, v in zip(todo, verdicts, strict=True):
                    seg = segments[i]
                    if v.decision == VERDICT_BLOCK:
                        decisions[i] = SegDecision(
                            VERDICT_BLOCK, "tier3", v.label, v.attack_prob, f"classifier: {v.label}", v
                        )
                    elif self.signatures.near_duplicate(v.embedding):
                        decisions[i] = SegDecision(
                            VERDICT_BLOCK,
                            "tier2",
                            "signature",
                            v.attack_prob,
                            "near-duplicate of a confirmed attack",
                            v,
                        )
                    elif v.decision == VERDICT_ABSTAIN:
                        decisions[i] = SegDecision(
                            VERDICT_ABSTAIN, "tier3", v.label, v.attack_prob, "inside the abstain band", v
                        )
                    else:
                        decisions[i] = SegDecision(VERDICT_ALLOW, "tier3", "benign", v.attack_prob, "benign", v)
                        self.cache.put(server, tool_key, seg.sha256, self.config.cache_ttl_s, self.clock())
        for i, seg in enumerate(segments):
            if decisions[i] is None:
                decisions[i] = decisions[seen[seg.sha256]]
        t2 = time.perf_counter()
        timings = {"tier2_ms": round((t1 - t0) * 1000, 3), "tier3_ms": round((t2 - t1) * 1000, 3)}
        return ScanOutcome([d for d in decisions if d is not None], segments, timings)

    def _warn_once(self, key: tuple[str, str], message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.log.emit("warning", "info", reason=message)

    # ------------------------------------------------------------------ blocks, holds, probes

    def _block(
        self,
        conn: Connection,
        msg_id: Any,
        kind: str,
        *,
        tier: str,
        reason: str,
        direction: str,
        session: Session | None,
        tool: str | None = None,
        method: str | None = None,
        cls: str | None = None,
        prob: float | None = None,
        segment: str | None = None,
        probe: str | None = None,
        respond: bool = True,
        **extra: Any,
    ) -> Decision:
        verdict = "hold" if kind == "hold" else "block"
        ref = self.log.emit(
            kind,
            verdict,
            tier=tier,
            cls=cls,
            prob=prob,
            server=conn.server,
            tool=tool,
            direction=direction,
            method=method,
            session=session.key if session else None,
            trace_id=(session.trace_ids[-1] if session and session.trace_ids else None),
            reason=reason,
            segment=segment,
            **extra,
        )
        self.stats["held" if verdict == "hold" else "blocked"] += 1
        if probe and session is not None:
            self._probe(session, conn.server, probe)
        to_host = [error_response(msg_id, stub_error(ref))] if respond else []
        return Decision(to_host=to_host, verdict=verdict, ref=ref)

    def _probe(self, session: Session, server: str, scope: str) -> None:
        now = self.clock()
        cfg = self.config
        self.store.add_probe("session", session.key, now)
        recent = self.store.count_probes("session", session.key, now - cfg.probe_strict_window_s)
        hourly = self.store.count_probes("session", session.key, now - cfg.probe_quarantine_window_s)
        if recent >= cfg.probe_strict_blocks and self.taint.escalate(session, "probe detection"):
            self.log.emit(
                "probe",
                "flag",
                session=session.key,
                server=server,
                reason=f"{recent} blocks in {cfg.probe_strict_window_s / 60:.0f} min: session escalated to strict",
            )
        if scope == "client" and hourly >= cfg.probe_quarantine_blocks and not session.quarantined:
            session.quarantined = True
            self.taint.escalate(session, "probe quarantine")
            self.log.emit(
                "quarantine",
                "flag",
                session=session.key,
                server=server,
                reason=f"{hourly} blocks in an hour: client session quarantined until reviewed",
            )
        if scope == "server":
            self.store.add_probe("server", server, now)
            n = self.store.count_probes("server", server, now - cfg.probe_quarantine_window_s)
            if n >= cfg.probe_quarantine_blocks and self.server_status(server) != STATUS_PROBE_QUARANTINED:
                self.store.set_server_status(server, STATUS_PROBE_QUARANTINED, f"{n} blocks in an hour")
                self.log.emit(
                    "quarantine",
                    "flag",
                    server=server,
                    reason=f"{n} blocks in an hour: server quarantined until reviewed",
                )

    def _segment_block(
        self,
        conn: Connection,
        msg_id: Any,
        outcome: ScanOutcome,
        *,
        direction: str,
        session: Session | None,
        tool: str,
        method: str | None,
        probe: str | None,
        respond: bool = True,
        what: str = "content",
    ) -> Decision:
        blocked = outcome.blocked
        assert blocked is not None
        seg, d = blocked
        kind = "injection" if d.decision == VERDICT_BLOCK else "unavailable"
        return self._block(
            conn,
            msg_id,
            kind,
            tier=d.tier,
            reason=f"{what} {seg.path}: {d.reason}",
            direction=direction,
            session=session,
            tool=tool,
            method=method,
            cls=d.label,
            prob=d.prob,
            segment=seg.text,
            probe=probe if d.decision == VERDICT_BLOCK else None,
            respond=respond,
            segment_sha=seg.sha256,
            timings=outcome.timings,
        )

    def _abstain(self, conn: Connection, outcome: ScanOutcome, session: Session, tool: str, direction: str) -> None:
        found = outcome.abstained
        if found is None:
            return
        seg, d = found
        self.taint.escalate(session, "abstain-band verdict")
        self.log.emit(
            "abstain",
            "abstain",
            tier="tier3",
            cls=d.label,
            prob=d.prob,
            server=conn.server,
            tool=tool,
            direction=direction,
            session=session.key,
            segment=seg.text,
            segment_sha=seg.sha256,
            reason=f"{seg.path}: inside the abstain band — forwarded, session escalated to strict",
        )

    # ------------------------------------------------------------------ host → server

    async def from_host(self, conn: Connection, message: Any, headers: dict[str, str] | None = None) -> Decision:
        self.stats["messages"] += 1
        t0 = time.perf_counter()
        problem = shape_problem(message)
        if problem:
            msg_id = (
                message.get("id") if isinstance(message, dict) and isinstance(message.get("id"), (str, int)) else None
            )
            return self._block(conn, msg_id, "policy", tier="tier1", reason=problem, direction="h2s", session=None)
        kind = kind_of(message)
        msg_id = message.get("id")
        method = message.get("method")

        if headers is not None and kind in (REQUEST, NOTIFICATION):
            schema = None
            if method == "tools/call":
                row = self.store.pins(conn.server).get(("tool", str(params_of(message).get("name"))))
                schema = row.current_def.get("inputSchema") if row else None
            mismatches = header_body_mismatches(headers, message, schema if isinstance(schema, dict) else None)
            if mismatches:
                ref = self.log.emit(
                    "policy",
                    "block",
                    tier="tier1",
                    server=conn.server,
                    direction="h2s",
                    method=method,
                    reason="HeaderMismatch: " + "; ".join(mismatches),
                )
                self.stats["blocked"] += 1
                err = make_error(P.HEADER_MISMATCH, "Header mismatch")
                return Decision(
                    to_host=[error_response(msg_id, err)] if kind == REQUEST else [],
                    verdict="block",
                    ref=ref,
                    http_status=400,
                )

        if kind in (RESPONSE, ERROR):
            conn.pending_s2h.pop(_id_key(msg_id), None)
            return Decision(to_server=[message])
        if kind == NOTIFICATION:
            return Decision(to_server=[message])

        params = params_of(message)
        meta = params.get("_meta")
        if (
            isinstance(meta, dict)
            and P.META_PROTOCOL_VERSION in meta
            and not isinstance(meta[P.META_PROTOCOL_VERSION], str)
        ):
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason="_meta protocolVersion is not a string",
                direction="h2s",
                session=None,
                method=method,
            )
        if method == "initialize":
            info = params.get("clientInfo")
            conn.client_info = info if isinstance(info, dict) else None
            conn.protocol_version = (
                params.get("protocolVersion") if isinstance(params.get("protocolVersion"), str) else None
            )
        session = self._session(conn, message)
        tier1_ms = (time.perf_counter() - t0) * 1000

        if method == "tools/call":
            return await self._host_tool_call(conn, session, message, tier1_ms)
        if method == "prompts/get":
            return await self._host_prompt_get(conn, session, message)
        if method == "resources/read":
            return await self._host_resource_read(conn, session, message)
        self._pend(conn, message, str(method), str(method), None, session)
        return Decision(to_server=[message])

    def _pend(
        self,
        conn: Connection,
        message: dict[str, Any],
        method: str,
        tool: str,
        resource: str | None,
        session: Session,
        privilege: str = READ,
    ) -> None:
        conn.pending_h2s[_id_key(message.get("id"))] = Pending(
            method, tool, resource, session.key, self.clock(), privilege
        )

    def _resource_of(self, args: Any, tool_def: dict[str, Any] | None, policy: ToolPolicy) -> str | None:
        if not isinstance(args, dict):
            return None
        props = (tool_def or {}).get("inputSchema", {}).get("properties", {}) if isinstance(tool_def, dict) else {}
        props = props if isinstance(props, dict) else {}
        for key, value in args.items():
            if isinstance(value, str) and field_type_for(str(key), props.get(key), policy, str(key)) in (
                FIELD_PATH,
                FIELD_URL,
            ):
                return value
        return None

    def _flow_granted(self, source: str, dest: str) -> bool:
        return self.store.flow_granted(source, dest, self.clock())

    async def _host_tool_call(
        self, conn: Connection, session: Session, message: dict[str, Any], tier1_ms: float
    ) -> Decision:
        msg_id = message.get("id")
        params = params_of(message)
        name = params.get("name")
        args = params.get("arguments")
        if not isinstance(name, str) or not name:
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason="tools/call without a tool name",
                direction="h2s",
                session=session,
                method="tools/call",
                probe="client",
            )
        row = self.store.pins(conn.server).get(("tool", name))
        exp = exposure(self.server_status(conn.server), "tool", row)
        if not exp.expose:
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason=f"tool not on the allowlist: {exp.reason}",
                direction="h2s",
                session=session,
                tool=name,
                method="tools/call",
                probe="client",
            )
        assert row is not None
        policy = self.config.tool_policy(conn.server, name)
        privilege = privilege_of(row.current_def, name, policy)
        t = time.perf_counter()
        t1 = check_tool_call(args, row.current_def, policy, conn.rules, local=conn.local)
        tier1_ms += (time.perf_counter() - t) * 1000
        if t1.schema_warning:
            self._warn_once((conn.server, name), f"{conn.server}/{name}: {t1.schema_warning}")
        if t1.block:
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason=t1.block,
                direction="h2s",
                session=session,
                tool=name,
                method="tools/call",
                probe="client",
                findings=[f.__dict__ for f in t1.findings],
            )

        ex = extract_arguments(args or {}, conn.server, name, self.config.min_free_text)
        outcome = await self.scan(ex.segments, conn.server, name, "argument")
        if outcome.blocked:
            return self._segment_block(
                conn,
                msg_id,
                outcome,
                direction="h2s",
                session=session,
                tool=name,
                method="tools/call",
                probe="client",
                what="argument",
            )
        self._abstain(conn, outcome, session, name, "h2s")

        resource = self._resource_of(args, row.current_def, policy)
        arg_strings = [(".".join(str(p) for p in loc), text) for loc, text in string_leaves(args or {})]
        calibrated = getattr(self.classifier, "embedding_cosine", None)
        arg_embs = [(seg.path, emb) for seg, emb in outcome.embeddings()] if calibrated else []
        decision = self.taint.check_call(
            session,
            conn.server,
            privilege,
            conn.remote,
            arg_strings,
            arg_embs,
            resource,
            flow_allowed=self._flow_granted,
            cosine_threshold=calibrated,
        )
        for o in decision.allowed_flows:
            self.log.emit(
                "flow",
                "allow",
                server=conn.server,
                tool=name,
                session=session.key,
                reason=f"trusted flow {o.record.server} -> {conn.server} used ({o.detail})",
            )
        ahash = args_hash(conn.server, name, args)
        if policy.taint_opt_out and decision.violation:
            self.log.emit(
                "flow",
                "allow",
                server=conn.server,
                tool=name,
                session=session.key,
                reason=f"taint opt-out for {conn.server}/{name}: overlap with {', '.join(decision.sources)} ignored",
            )
            decision.overlaps.clear()
        granted = False
        if decision.violation:
            if self.store.consume_grant(session.key, conn.server, name, ahash, self.clock()):
                granted = True
                self.log.emit(
                    "override",
                    "allow",
                    server=conn.server,
                    tool=name,
                    session=session.key,
                    reason="Allow once grant used for a taint block",
                )
            else:
                o = decision.overlaps[0]
                return self._block(
                    conn,
                    msg_id,
                    "taint",
                    tier="taint",
                    direction="h2s",
                    session=session,
                    tool=name,
                    method="tools/call",
                    probe="client",
                    reason=f"argument {o.argument} carries tainted content: {o.detail}",
                    sources=decision.sources,
                    args_hash=ahash,
                    privilege=privilege,
                    actions=["allow_once", "allow_flow"],
                )
        hold = None
        if policy.requires_approval:
            hold = "the tool requires approval for every call"
        elif session.quarantined and privilege != READ:
            hold = "session quarantined by probe detection"
        else:
            hold = self.taint.needs_hold(session, privilege, shell_mode=policy.mode == "shell", annotated=t1.annotated)
        if hold and not granted:
            if self.store.consume_grant(session.key, conn.server, name, ahash, self.clock()):
                self.log.emit(
                    "override",
                    "allow",
                    server=conn.server,
                    tool=name,
                    session=session.key,
                    reason=f"Allow once grant used ({hold})",
                )
            else:
                return self._block(
                    conn,
                    msg_id,
                    "hold",
                    tier="taint",
                    direction="h2s",
                    session=session,
                    tool=name,
                    method="tools/call",
                    reason=f"held for approval: {hold}",
                    args_hash=ahash,
                    privilege=privilege,
                    actions=["allow_once"],
                )
        self._pend(conn, message, "tools/call", name, resource, session, privilege)
        if t1.annotated:
            self.log.emit(
                "annotation",
                "allow",
                server=conn.server,
                tool=name,
                session=session.key,
                tier="tier1",
                reason="; ".join(f"{f.path}: {f.kind} {f.detail}" for f in t1.findings if not f.block),
            )
        return Decision(to_server=[message])

    async def _host_prompt_get(self, conn: Connection, session: Session, message: dict[str, Any]) -> Decision:
        msg_id = message.get("id")
        params = params_of(message)
        name = str(params.get("name", ""))
        row = self.store.pins(conn.server).get(("prompt", name))
        exp = exposure(self.server_status(conn.server), "prompt", row)
        if not exp.expose:
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason=f"prompt not on the allowlist: {exp.reason}",
                direction="h2s",
                session=session,
                tool=name,
                method="prompts/get",
                probe="client",
            )
        args = params.get("arguments")
        ex = extract_arguments(args or {}, conn.server, name, self.config.min_free_text)
        outcome = await self.scan(ex.segments, conn.server, f"prompt:{name}", "argument")
        if outcome.blocked:
            return self._segment_block(
                conn,
                msg_id,
                outcome,
                direction="h2s",
                session=session,
                tool=name,
                method="prompts/get",
                probe="client",
                what="argument",
            )
        if conn.remote:
            arg_strings = [(".".join(str(p) for p in loc), t) for loc, t in string_leaves(args or {})]
            decision = self.taint.check_call(
                session, conn.server, READ, True, arg_strings, flow_allowed=self._flow_granted
            )
            if decision.violation:
                return self._block(
                    conn,
                    msg_id,
                    "taint",
                    tier="taint",
                    direction="h2s",
                    session=session,
                    tool=name,
                    method="prompts/get",
                    probe="client",
                    sources=decision.sources,
                    reason=f"prompt arguments carry tainted content: {decision.overlaps[0].detail}",
                    args_hash=args_hash(conn.server, name, args),
                    actions=["allow_once", "allow_flow"],
                )
        self._pend(conn, message, "prompts/get", name, None, session)
        return Decision(to_server=[message])

    async def _host_resource_read(self, conn: Connection, session: Session, message: dict[str, Any]) -> Decision:
        msg_id = message.get("id")
        uri = params_of(message).get("uri")
        if not isinstance(uri, str) or not uri:
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason="resources/read without a uri",
                direction="h2s",
                session=session,
                method="resources/read",
                probe="client",
            )
        if conn.remote:
            decision = self.taint.check_call(
                session, conn.server, READ, True, [("uri", uri)], flow_allowed=self._flow_granted
            )
            if decision.violation:
                return self._block(
                    conn,
                    msg_id,
                    "taint",
                    tier="taint",
                    direction="h2s",
                    session=session,
                    tool=uri,
                    method="resources/read",
                    probe="client",
                    sources=decision.sources,
                    reason=f"resource URI carries tainted content: {decision.overlaps[0].detail}",
                    args_hash=args_hash(conn.server, uri, None),
                    actions=["allow_once", "allow_flow"],
                )
        self._pend(conn, message, "resources/read", uri, uri, session)
        return Decision(to_server=[message])

    # ------------------------------------------------------------------ server → host

    async def from_server(self, conn: Connection, message: Any) -> Decision:
        self.stats["messages"] += 1
        problem = shape_problem(message)
        if problem:
            self.log.emit(
                "policy",
                "block",
                tier="tier1",
                server=conn.server,
                direction="s2h",
                origin="server",
                reason=f"server sent {problem}; dropped",
            )
            return Decision(verdict="block")
        kind = kind_of(message)
        if kind in (RESPONSE, ERROR):
            pending = conn.pending_h2s.pop(_id_key(message.get("id")), None)
            session = self.taint.sessions.get(pending.session_key) if pending else None
            session = session or self._conn_session(conn)
            method = pending.method if pending else "unknown"
            if kind == ERROR:
                return await self._server_error(conn, message, pending, session)
            result = message.get("result")
            if method in P.PINNED_LIST_METHODS:
                return await self._server_list(conn, message, method, session)
            if (
                method in RESULT_METHODS
                and isinstance(result, dict)
                and result.get("resultType") == P.RESULT_INPUT_REQUIRED
            ):
                return await self._server_mrtr(conn, message, pending, session)
            if method in ("initialize", "server/discover"):
                return await self._server_init(conn, message, method, session)
            if method == "resources/list":
                return await self._server_resource_list(conn, message, session)
            return await self._server_content(conn, message, pending, session, method)
        if kind == REQUEST:
            return await self._server_request(conn, message)
        return await self._server_notification(conn, message)

    def _spotlight_on(self, server: str, policy: ToolPolicy) -> bool:
        server_policy = self.config.servers.get(server)
        if server_policy is not None and server_policy.spotlight is False:
            return False
        return self.config.spotlight and policy.spotlight

    async def _server_content(
        self, conn: Connection, message: dict[str, Any], pending: Pending | None, session: Session, method: str
    ) -> Decision:
        msg_id = message.get("id")
        result = message.get("result")
        tool = pending.tool if pending else method
        privilege = pending.privilege if pending else READ
        policy = self.config.tool_policy(conn.server, tool) if method == "tools/call" else ToolPolicy()
        spot = method in RESULT_METHODS and self._spotlight_on(conn.server, policy)
        if method == "tools/call" and not policy.spotlight:
            self._warn_once(
                ("spotlight-off", f"{conn.server}/{tool}"),
                f"Spotlighting disabled for {conn.server}/{tool} by configuration",
            )
        ex = extract_result(method, result, conn.server, tool, spotlight=spot)

        originals = [s for s in ex.segments if not s.derived]
        total = sum(len(s.raw.encode("utf-8")) for s in originals)
        cap = (
            self.store.tool_cap(conn.server, tool)
            or policy.result_cap
            or (self.config.result_cap_read if privilege == READ else self.config.result_cap_default)
        )
        dos = dos_check(
            total,
            "\n".join(s.raw for s in originals),
            cap=cap,
            read_class=privilege == READ,
            min_size=self.config.dos_min_size,
            max_ratio=self.config.dos_repetition_ratio,
        )
        if dos.action == "reject":
            return self._block(
                conn,
                msg_id,
                "dos",
                tier="tier2",
                reason=f"Context-DoS guard: {dos.reason}",
                direction="s2h",
                session=session,
                tool=tool,
                method=method,
                probe="server",
                size=total,
                cap=cap,
                actions=["raise_cap"],
            )
        if dos.action == "truncate":
            self._truncate(result, originals, cap, total)
            ex = extract_result(method, result, conn.server, tool, spotlight=spot)
            self.log.emit(
                "truncate",
                "flag",
                tier="tier2",
                server=conn.server,
                tool=tool,
                direction="s2h",
                session=session.key,
                reason=f"read-class result truncated: {dos.reason}",
                size=total,
                cap=cap,
                actions=["raise_cap"],
            )

        outcome = await self.scan(ex.segments, conn.server, tool, "content")
        if outcome.blocked:
            return self._segment_block(
                conn, msg_id, outcome, direction="s2h", session=session, tool=tool, method=method, probe="server"
            )
        self._abstain(conn, outcome, session, tool, "s2h")
        texts = [s.raw for s in ex.segments if not s.derived]
        embs = [e for _, e in outcome.embeddings()]
        self.taint.record(
            session,
            conn.server,
            tool,
            pending.resource if pending else None,
            texts,
            [b.sha256 for b in ex.blobs],
            embs,
            session.trace_ids[-1] if session.trace_ids else None,
        )
        if texts and self.server_status(conn.server) in (STATUS_QUARANTINED, STATUS_PROBE_QUARANTINED):
            self.taint.escalate(session, f"result from quarantined server {conn.server}")
        if spot and isinstance(result, dict):
            spotlight.apply(result, ex.segments, conn.server, SOURCE_KIND.get(method, "tool"), tool)
        if method in RESULT_METHODS:
            self.log.emit(
                "traffic",
                "allow",
                server=conn.server,
                tool=tool,
                method=method,
                direction="s2h",
                session=session.key,
                timings=outcome.timings,
                segments=len(ex.segments),
                latency_ms=round((self.clock() - pending.started) * 1000, 1) if pending else None,
            )
        return Decision(to_host=[message])

    @staticmethod
    def _truncate(result: Any, segments: list[Segment], cap: int, total: int) -> None:
        budget = cap
        note = f"\n[MCP-Shield: output truncated — {total} bytes exceeded the {cap}-byte cap]"
        for seg in segments:
            raw = seg.raw.encode("utf-8")
            if len(raw) <= budget:
                budget -= len(raw)
                continue
            kept = raw[:budget].decode("utf-8", errors="ignore")
            set_at(result, seg.loc, kept + note)
            budget = 0
            note = ""

    async def _server_list(self, conn: Connection, message: dict[str, Any], method: str, session: Session) -> Decision:
        kind = P.PINNED_LIST_METHODS[method]
        key = LIST_KEYS[kind]
        result = message.get("result")
        if not isinstance(result, dict):
            return self._block(
                conn,
                message.get("id"),
                "policy",
                tier="tier1",
                reason=f"{method} result is not an object",
                direction="s2h",
                session=session,
                method=method,
            )
        items = result.get(key)
        items = items if isinstance(items, list) else []
        now = self.clock()
        ttl = self.config.cache_ttl_s
        if isinstance(result.get("ttlMs"), (int, float)) and result["ttlMs"] >= 0:
            ttl = min(ttl, result["ttlMs"] / 1000)
        pins = self.store.pins(conn.server)
        server_row = self.store.server(conn.server)
        status = str(server_row["status"]) if server_row else STATUS_QUARANTINED
        first_contact = not pins and status == STATUS_QUARANTINED

        parsed: list[tuple[dict[str, Any], str, dict[str, Any], str]] = []
        for item in items:
            if isinstance(item, dict) and item_name(kind, item):
                defn = canonical_def(kind, item)
                parsed.append((item, item_name(kind, item), defn, def_hash(defn)))
        if first_contact:
            match = matches_known_manifest({h for *_, h in parsed}, self.known_manifests)
            if match:
                self.log.emit("pin", "allow", server=conn.server, reason=f"matches the published manifest '{match}'")
            else:
                self.log.emit(
                    "quarantine",
                    "flag",
                    server=conn.server,
                    method=method,
                    reason="first contact: server quarantined until its manifest is reviewed",
                    actions=["accept_pins"],
                )
        else:
            match = None

        # scan whatever is new or stale, in one batch
        to_scan: list[int] = []
        reuse: dict[int, tuple[str, str, float | None]] = {}
        for i, (_item, name, _defn, h) in enumerate(parsed):
            row = pins.get((kind, name))
            if (
                row
                and row.current_hash == h
                and row.scan_decision in (VERDICT_ALLOW, VERDICT_ABSTAIN, VERDICT_BLOCK)
                and row.scanned
                and now - row.scanned < ttl
            ):
                reuse[i] = (row.scan_decision, row.scan_label or "benign", row.scan_prob)
            else:
                to_scan.append(i)
        seg_owner: list[int] = []
        segments: list[Segment] = []
        for i in to_scan:
            for def_seg in extract_definition(kind, parsed[i][0], conn.server).segments:
                segments.append(def_seg)
                seg_owner.append(i)
        outcome = await self.scan(segments, conn.server, f"{kind}-description", "description") if segments else None
        scanned: dict[int, tuple[str, str, float | None, Segment | None]] = {
            i: (VERDICT_ALLOW, "benign", 0.0, None) for i in to_scan
        }
        if outcome is not None:
            for scan_seg, d, owner in zip(outcome.segments, outcome.decisions, seg_owner, strict=True):
                current = scanned[owner]
                if SEVERITY[d.decision] > SEVERITY[current[0]] or (
                    d.decision == current[0] and (d.prob or 0) > (current[2] or 0)
                ):
                    scanned[owner] = (d.decision, d.label, d.prob, scan_seg)

        exposed: list[dict[str, Any]] = []
        tampered = False
        for i, (item, name, defn, h) in enumerate(parsed):
            row = pins.get((kind, name))
            privilege = privilege_of(defn, name, self.config.tool_policy(conn.server, name)) if kind == "tool" else None
            seg: Segment | None
            if i in reuse:
                decision, label, prob = reuse[i]
                seg = None
                fresh = False
            else:
                decision, label, prob, seg = scanned[i]
                fresh = True
            self.store.upsert_pin(
                conn.server,
                kind,
                name,
                h,
                defn,
                privilege,
                (decision, label, prob or 0.0, now if fresh else (row.scanned or now) if row else now),
                accept=bool(match),
            )
            updated = self.store.pins(conn.server).get((kind, name))
            exp = exposure(STATUS_ACCEPTED if match else status, kind, updated)
            if exp.expose:
                exposed.append(item)
                if fresh and decision == VERDICT_ABSTAIN and seg is not None:
                    self.taint.escalate(session, "abstain-band verdict on a description")
                    self.log.emit(
                        "abstain",
                        "abstain",
                        tier="tier3",
                        cls=label,
                        prob=prob,
                        server=conn.server,
                        tool=name,
                        direction="s2h",
                        session=session.key,
                        segment=seg.text,
                        reason=f"{kind} description inside the abstain band — exposed, session strict",
                    )
                continue
            new_change = row is None or row.current_hash != h or fresh
            if exp.event == "tamper" and new_change:
                tampered = True
                self.log.emit(
                    "tamper",
                    "block",
                    tier="pin",
                    server=conn.server,
                    tool=name,
                    direction="s2h",
                    method=method,
                    reason=f"{kind} '{name}': {exp.reason} — withheld",
                    diff=diff_definitions(row.accepted_def if row else None, defn),
                    actions=["accept_pins"],
                )
            elif exp.event == "injection" and fresh:
                self.log.emit(
                    "injection",
                    "block",
                    tier="tier3" if seg is None or label != "signature" else "tier2",
                    cls=label,
                    prob=prob,
                    server=conn.server,
                    tool=name,
                    direction="s2h",
                    method=method,
                    segment=seg.text if seg else None,
                    reason=f"{kind} '{name}' description flagged ({label}) — withheld",
                )
                self._probe(session, conn.server, "server")
            elif exp.event == "scan_error" and fresh:
                self.log.emit(
                    "unavailable",
                    "block",
                    tier="tier3",
                    server=conn.server,
                    tool=name,
                    direction="s2h",
                    method=method,
                    reason=f"{kind} '{name}' withheld: classifier unavailable (fail closed)",
                )
        if tampered:
            self.cache.invalidate_server(conn.server)
        if match:
            self.store.accept_pins(conn.server)
        result[key] = exposed
        self.log.emit(
            "traffic",
            "allow",
            server=conn.server,
            method=method,
            direction="s2h",
            session=session.key,
            reason=f"{len(exposed)} of {len(items)} {key} exposed",
            timings=outcome.timings if outcome else None,
        )
        return Decision(to_host=[message])

    async def _server_init(self, conn: Connection, message: dict[str, Any], method: str, session: Session) -> Decision:
        result = message.get("result")
        if not isinstance(result, dict):
            return Decision(to_host=[message])
        if method == "initialize" and isinstance(result.get("protocolVersion"), str):
            conn.protocol_version = result["protocolVersion"]
        ex = extract_result(method, result, conn.server, "instructions")
        if ex.segments:
            outcome = await self.scan(ex.segments, conn.server, "instructions", "description")
            if outcome.blocked:
                result.pop("instructions", None)
                seg, d = outcome.blocked
                self.log.emit(
                    "injection" if d.decision == VERDICT_BLOCK else "unavailable",
                    "block",
                    tier=d.tier,
                    cls=d.label,
                    prob=d.prob,
                    server=conn.server,
                    direction="s2h",
                    method=method,
                    session=session.key,
                    segment=seg.text,
                    reason=f"server instructions removed: {d.reason}",
                )
            else:
                self._abstain(conn, outcome, session, "instructions", "s2h")
        return Decision(to_host=[message])

    async def _server_resource_list(self, conn: Connection, message: dict[str, Any], session: Session) -> Decision:
        result = message.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("resources"), list):
            return Decision(to_host=[message])
        ex = extract_result("resources/list", result, conn.server, "resources")
        if not ex.segments:
            return Decision(to_host=[message])
        outcome = await self.scan(ex.segments, conn.server, "resource-description", "description")
        bad = {
            seg.loc[1]
            for seg, d in zip(outcome.segments, outcome.decisions, strict=True)
            if d.decision in (VERDICT_BLOCK, SCAN_ERROR)
        }
        if bad:
            worst = outcome.blocked
            assert worst is not None
            seg, d = worst
            self.log.emit(
                "injection" if d.decision == VERDICT_BLOCK else "unavailable",
                "block",
                tier=d.tier,
                cls=d.label,
                prob=d.prob,
                server=conn.server,
                direction="s2h",
                method="resources/list",
                session=session.key,
                segment=seg.text,
                reason=f"{len(bad)} resource(s) withheld: {d.reason}",
            )
            result["resources"] = [r for i, r in enumerate(result["resources"]) if i not in bad]
        return Decision(to_host=[message])

    async def _server_error(
        self, conn: Connection, message: dict[str, Any], pending: Pending | None, session: Session
    ) -> Decision:
        err = message.get("error")
        tool = pending.tool if pending else "unknown"
        if isinstance(err, dict) and is_stub_message(err.get("message")):
            self.log.emit(
                "server_error",
                "info",
                origin="server",
                server=conn.server,
                tool=tool,
                direction="s2h",
                session=session.key,
                reason="the server returned an MCP-Shield-style block message; its reference resolves in "
                "the server operator's log, not here",
            )
        ex = extract_error(err, conn.server, tool)
        if not ex.segments:
            return Decision(to_host=[message])
        outcome = await self.scan(ex.segments, conn.server, f"error:{tool}", "content")
        if outcome.blocked:
            decision = self._segment_block(
                conn,
                message.get("id"),
                outcome,
                direction="s2h",
                session=session,
                tool=tool,
                method=pending.method if pending else None,
                probe="server",
                what="server error text",
            )
            return decision
        self._abstain(conn, outcome, session, tool, "s2h")
        return Decision(to_host=[message])

    def _url_problems(self, url: Any) -> tuple[list[str], str | None]:
        if not isinstance(url, str):
            return ([], None)
        problems = check_url(url, ToolPolicy(url_schemes=["https"]))
        try:
            host = urlsplit(url).hostname
        except ValueError:
            host = None
        return problems, host

    async def _server_mrtr(
        self, conn: Connection, message: dict[str, Any], pending: Pending | None, session: Session
    ) -> Decision:
        msg_id = message.get("id")
        result = message["result"]
        tool = pending.tool if pending else "unknown"
        method = pending.method if pending else None
        requests = result.get("inputRequests")
        if not isinstance(requests, dict):
            return self._block(
                conn,
                msg_id,
                "policy",
                tier="tier1",
                reason="input_required result without an inputRequests map",
                direction="s2h",
                session=session,
                tool=tool,
                method=method,
            )
        per_request: list[tuple[list[Segment], str | None]] = []
        all_segments: list[Segment] = []
        for rid, req in requests.items():
            if not isinstance(req, dict) or not isinstance(req.get("method"), str):
                return self._block(
                    conn,
                    msg_id,
                    "policy",
                    tier="tier1",
                    reason=f"malformed input request {rid!r}",
                    direction="s2h",
                    session=session,
                    tool=tool,
                    method=method,
                )
            req_params = req.get("params")
            ex = extract_input_request(req["method"], req_params, conn.server, ("inputRequests", rid, "params"))
            url_host = None
            if req["method"] == "elicitation/create" and isinstance(req_params, dict) and "url" in req_params:
                problems, url_host = self._url_problems(req_params.get("url"))
                if problems:
                    return self._block(
                        conn,
                        msg_id,
                        "policy",
                        tier="tier1",
                        direction="s2h",
                        session=session,
                        tool=tool,
                        method=method,
                        probe="server",
                        reason=f"URL-mode elicitation link rejected: {'; '.join(problems)}",
                    )
            per_request.append((ex.segments, url_host))
            all_segments.extend(ex.segments)
        outcome = await self.scan(all_segments, conn.server, f"input-request:{tool}", "content")
        if outcome.blocked:
            return self._segment_block(
                conn,
                msg_id,
                outcome,
                direction="s2h",
                session=session,
                tool=tool,
                method=method,
                probe="server",
                what="input request",
            )
        self._abstain(conn, outcome, session, tool, "s2h")
        texts = [s.raw for s in all_segments if not s.derived]
        self.taint.record(session, conn.server, tool, None, texts, [], [e for _, e in outcome.embeddings()])
        for segments, url_host in per_request:
            spotlight.add_banner(result, segments, conn.server, url_host)
        self.log.emit(
            "traffic",
            "allow",
            server=conn.server,
            tool=tool,
            method=method,
            direction="s2h",
            session=session.key,
            reason=f"input_required with {len(requests)} request(s); banner added",
            timings=outcome.timings,
        )
        return Decision(to_host=[message])

    async def _server_request(self, conn: Connection, message: dict[str, Any]) -> Decision:
        method = str(message.get("method"))
        msg_id = message.get("id")
        session = self._conn_session(conn)
        if method in ("sampling/createMessage", "elicitation/create"):
            params = params_of(message)
            ex = extract_input_request(method, params, conn.server, ())
            url_host = None
            if method == "elicitation/create" and "url" in params:
                problems, url_host = self._url_problems(params.get("url"))
                if problems:
                    ref = self.log.emit(
                        "policy",
                        "block",
                        tier="tier1",
                        server=conn.server,
                        method=method,
                        direction="s2h",
                        session=session.key,
                        reason=f"URL-mode elicitation link rejected: {'; '.join(problems)}; declined",
                    )
                    self._probe(session, conn.server, "server")
                    return Decision(to_server=[_decline(method, msg_id)], verdict="block", ref=ref)
            outcome = await self.scan(ex.segments, conn.server, method, "content")
            if outcome.blocked:
                decision = self._segment_block(
                    conn,
                    msg_id,
                    outcome,
                    direction="s2h",
                    session=session,
                    tool=method,
                    method=method,
                    probe="server",
                    respond=False,
                    what="server request",
                )
                decision.to_server = [_decline(method, msg_id)]
                return decision
            self._abstain(conn, outcome, session, method, "s2h")
            self.taint.record(
                session,
                conn.server,
                method,
                None,
                [s.raw for s in ex.segments if not s.derived],
                [],
                [e for _, e in outcome.embeddings()],
            )
            spotlight.add_banner(params, ex.segments, conn.server, url_host)
        conn.pending_s2h[_id_key(msg_id)] = method
        return Decision(to_host=[message])

    async def _server_notification(self, conn: Connection, message: dict[str, Any]) -> Decision:
        method = str(message.get("method"))
        if method in P.LIST_CHANGED_NOTIFICATIONS:
            return Decision(to_host=[message])
        ex = extract_notification(method, message.get("params"), conn.server)
        if not ex.segments:
            return Decision(to_host=[message])
        session = self._conn_session(conn)
        outcome = await self.scan(ex.segments, conn.server, method, "content")
        if outcome.blocked:
            decision = self._segment_block(
                conn,
                None,
                outcome,
                direction="s2h",
                session=session,
                tool=method,
                method=method,
                probe="server",
                respond=False,
                what="notification",
            )
            return decision  # dropped: notifications have no response
        self._abstain(conn, outcome, session, method, "s2h")
        return Decision(to_host=[message])

    # ------------------------------------------------------------------ allowlist (HTTP pre-body check)

    def name_allowed(self, server: str, method: str, name: str) -> bool:
        if method != "tools/call":
            return True
        row = self.store.pins(server).get(("tool", name))
        return exposure(self.server_status(server), "tool", row).expose

    # ------------------------------------------------------------------ dashboard / CLI actions

    def action(self, ref: str, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply a one-click action to the event ``ref``; every action is logged and reversible."""
        params = params or {}
        event = self.store.event(ref)
        if event is None:
            raise KeyError(f"no event with reference {ref}")
        now = self.clock()
        server, tool, session_key = event.get("server"), event.get("tool"), event.get("session")
        kind = event.get("kind")
        note: str
        if action == "allow_content":
            if (
                kind not in ("injection", "abstain")
                or not event.get("segment_sha")
                or event.get("cls") == "unavailable"
            ):
                raise ValueError("Allow this exact content applies to classifier or signature verdicts only")
            tool_key = str(tool)
            self.store.add_allow_content(
                str(server), tool_key, event["segment_sha"], now + self.config.content_allow_ttl_s, ref
            )
            self.cache.put(str(server), tool_key, event["segment_sha"], self.config.content_allow_ttl_s, now)
            note = f"exact content allowed for {server}/{tool} for {self.config.content_allow_ttl_s / 3600:.0f} h"
        elif action == "allow_once":
            if kind not in ("taint", "hold") or not event.get("args_hash"):
                raise ValueError("Allow once applies to taint blocks and held calls")
            self.store.add_grant(
                str(session_key), str(server), str(tool), event["args_hash"], now + self.config.grant_ttl_s, ref
            )
            minutes = self.config.grant_ttl_s / 60
            note = f"single-use grant for the identical call, valid {minutes:.0f} min — ask the agent to retry"
        elif action == "allow_flow":
            if kind != "taint" or not event.get("sources"):
                raise ValueError("Allow this flow applies to taint blocks")
            for source in event["sources"]:
                self.store.add_flow_grant(source, str(server), now + self.config.flow_grant_ttl_s, ref)
            hours = self.config.flow_grant_ttl_s / 3600
            note = f"flow {', '.join(event['sources'])} -> {server} allowed for {hours:.0f} h"
        elif action == "raise_cap":
            if kind not in ("dos", "truncate"):
                raise ValueError("Raise this tool's result cap applies to DoS-guard items")
            cap = int(params.get("cap") or max(int(event.get("size") or 0) * 2, int(event.get("cap") or 0) * 2))
            self.store.set_tool_cap(str(server), str(tool), cap)
            note = f"result cap for {server}/{tool} raised to {cap} bytes"
        elif action == "mark_benign":
            if kind != "abstain":
                raise ValueError("Mark benign applies to abstain-band items")
            session = self.taint.sessions.get(str(session_key))
            if session is not None:
                self.taint.reset(session)
            raw = self.log.raw_segment(ref) or event.get("segment") or ""
            self.store.add_review("hard_negative", raw, "benign", None, ref)
            if event.get("segment_sha"):
                self.cache.put(str(server), str(tool), event["segment_sha"], self.config.content_allow_ttl_s, now)
            note = "marked benign: session returned to standard, segment queued as a hard negative"
        elif action == "mark_attack":
            raw = self.log.raw_segment(ref) or event.get("segment")
            if not raw:
                raise ValueError("the segment is no longer available (retention or engine restart)")
            sig = self.signatures.add(raw)
            self.store.add_signature(sig.sha256, raw, None, ref)
            self.store.add_review("attack", raw, event.get("cls"), None, ref)
            note = "added to the local signature set and queued as an attack example"
        elif action == "report_fp":
            if event.get("tier") not in ("tier3", "tier2"):
                raise ValueError("Report false positive applies to classifier verdicts")
            raw = self.log.raw_segment(ref) or event.get("segment") or ""
            self.store.add_review("false_positive", raw, "benign", None, ref)
            note = "queued for the next training round's hard negatives"
        elif action == "accept_pins":
            n = self.accept_pins(str(server))
            note = f"{n} definitions accepted for {server}"
        else:
            raise ValueError(f"unknown action {action!r}")
        self.store.set_event_status(ref, "resolved")
        self.log.emit(
            "override",
            "allow",
            server=server,
            tool=tool,
            session=session_key,
            reason=f"{action} on {ref}: {note}",
            target=ref,
        )
        return {"ok": True, "note": note}

    def accept_pins(self, server: str, items: list[tuple[str, str]] | None = None) -> int:
        n = self.store.accept_pins(server, items)
        self.cache.invalidate_server(server)
        self.log.emit("pin", "allow", server=server, reason=f"pins accepted by the user ({n} definitions)")
        return n

    def reset_session(self, key: str) -> bool:
        session = self.taint.sessions.get(key)
        if session is None:
            return False
        self.taint.reset(session)
        self.log.emit("override", "allow", session=key, reason="session reset to standard by the user")
        return True

    def release_server(self, server: str) -> None:
        self.store.set_server_status(
            server,
            STATUS_ACCEPTED if any(r.accepted_hash for r in self.store.pins(server).values()) else STATUS_QUARANTINED,
        )
        self.log.emit("override", "allow", server=server, reason="probe quarantine released by the user")

    # ------------------------------------------------------------------ views

    def sessions_view(self) -> list[dict[str, Any]]:
        out = []
        for s in self.taint.sessions.values():
            live = self.taint.live(s.group)
            out.append(
                {
                    "key": s.key,
                    "client": s.client,
                    "host": s.host,
                    "policy": s.policy,
                    "strict_reasons": s.strict_reasons,
                    "quarantined": s.quarantined,
                    "last_seen": s.last_seen,
                    "tainted": bool(live),
                    "live_records": len(live),
                    "sources": sorted({r.server for r in live}),
                    "trace_ids": s.trace_ids[-3:],
                }
            )
        return sorted(out, key=lambda s: -s["last_seen"])

    def pins_view(self, server: str) -> list[dict[str, Any]]:
        status = self.server_status(server)
        out = []
        for (kind, name), row in sorted(self.store.pins(server).items()):
            exp = exposure(status, kind, row)
            out.append(
                {
                    "kind": kind,
                    "name": name,
                    "privilege": row.privilege,
                    "accepted": row.accepted,
                    "changed": row.changed,
                    "exposed": exp.expose,
                    "reason": exp.reason,
                    "scan": {"decision": row.scan_decision, "label": row.scan_label, "prob": row.scan_prob},
                    "hash": row.current_hash,
                    "diff": diff_definitions(row.accepted_def, row.current_def) if row.changed else [],
                    "description": str(row.current_def.get("description", ""))[:2000],
                }
            )
        return out

    def get_at_safe(self, obj: Any, loc: tuple[Any, ...]) -> Any:  # pragma: no cover - debugging helper
        try:
            return get_at(obj, loc)
        except (KeyError, IndexError, TypeError):
            return None
