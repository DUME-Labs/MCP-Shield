import numpy as np

from mcp_shield.config import SecurityConfig
from mcp_shield.pinning import (
    STATUS_ACCEPTED,
    STATUS_PROBE_QUARANTINED,
    STATUS_QUARANTINED,
    canonical_def,
    def_hash,
    diff_definitions,
    exposure,
    matches_known_manifest,
)
from mcp_shield.taint import TaintEngine, args_hash

SECRET = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
DOC = f"# deployment notes\nThe staging cluster uses {SECRET} for uploads.\nRotate quarterly."


class TestStore:
    def test_pins_accept_and_change(self, store):
        store.touch_server("fs", "stdio", False, "npx server")
        store.upsert_pin("fs", "tool", "read_file", "h1", {"name": "read_file"}, "read", ("allow", "benign", 0.01, 1.0))
        row = store.pins("fs")[("tool", "read_file")]
        assert row.accepted_hash is None and not row.accepted and not row.changed
        assert store.accept_pins("fs") == 1
        assert store.server("fs")["status"] == "accepted"
        store.upsert_pin(
            "fs",
            "tool",
            "read_file",
            "h2",
            {"name": "read_file", "description": "x"},
            "read",
            ("allow", "benign", 0.01, 2.0),
        )
        row = store.pins("fs")[("tool", "read_file")]
        assert row.changed and row.accepted_def == {"name": "read_file"}

    def test_events_round_trip(self, store):
        store.add_event(
            {
                "ref": "0000abcd",
                "ts": 1.0,
                "kind": "taint",
                "verdict": "block",
                "server": "s",
                "sources": ["a"],
                "args_hash": "x",
            }
        )
        e = store.event("0000abcd")
        assert e["sources"] == ["a"] and e["origin"] == "proxy" and e["status"] == "open"
        assert store.ref_exists("0000abcd")
        assert [r["ref"] for r in store.events(verdicts=["block"])] == ["0000abcd"]
        assert store.events(query="0000ab")

    def test_grant_single_use_and_expiry(self, store):
        store.add_grant("sess", "srv", "tool", "h", expires=100.0, ref=None)
        assert not store.consume_grant("sess", "srv", "tool", "other", 50.0)
        assert store.consume_grant("sess", "srv", "tool", "h", 50.0)
        assert not store.consume_grant("sess", "srv", "tool", "h", 50.0), "single use"
        store.add_grant("sess", "srv", "tool", "h", expires=100.0, ref=None)
        assert not store.consume_grant("sess", "srv", "tool", "h", 101.0), "expired"

    def test_flows_and_purge(self, store):
        store.add_flow_grant("fs", "gh", 100.0, None)
        assert store.flow_granted("fs", "gh", 50) and not store.flow_granted("gh", "fs", 50)
        store.add_event({"ref": "1", "ts": 0.0, "kind": "x", "verdict": "block", "segment": "secret"})
        store.purge(now=8 * 86400, retention_days=7, taint_horizon_s=1800)
        assert store.event("1")["segment"] is None
        assert not store.flow_granted("fs", "gh", 8 * 86400)


class TestTaint:
    def engine(self, clock, store=None, **kw):
        return TaintEngine(SecurityConfig(**kw), store, clock)

    def session(self, t, host="host:1", client="claude"):
        return t.session_for(None, client, "1.0", host)

    def test_session_key_is_not_the_server(self, clock):
        t = self.engine(clock)
        a, b = self.session(t), self.session(t)
        assert a is b
        assert self.session(t, host="other").key != a.key
        assert self.session(t, host="other").group == a.group, "same client name: same taint group"

    def test_containment_blocks_exfiltration(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "filesystem", "read_file", "/work/deploy.md", [DOC], [], [])
        d = t.check_call(s, "mail", "egress", False, [("body", f"FYI the key is {SECRET}")])
        assert d.violation and d.sources == ["filesystem"] and d.overlaps[0].how == "shared-run"

    def test_short_overlap_is_not_a_flow(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "filesystem", "read_file", "/a", [DOC], [], [])
        assert not t.check_call(s, "mail", "egress", False, [("body", "Rotate quarterly please")]).violation

    def test_read_class_local_call_not_checked(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "fs", "read_file", "/a", [DOC], [], [])
        assert not t.check_call(s, "other", "read", False, [("q", SECRET)]).checked

    def test_rule_b_remote_server_any_class(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "fs", "read_file", "/a", [DOC], [], [])
        d = t.check_call(s, "remote-wiki", "read", True, [("id", SECRET)])
        assert d.checked and d.violation
        d = t.check_call(s, "fs", "read", True, [("id", SECRET)])
        assert not d.violation, "rule (b) only concerns servers other than the source"

    def test_write_back_exempt(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "fs", "read_file", "/work/deploy.md", [DOC], [], [])
        same = t.check_call(s, "fs", "write", False, [("content", DOC + " edited")], resource="/work/deploy.md")
        assert not same.violation and same.exempt_write_back == 1
        other = t.check_call(s, "fs", "write", False, [("content", DOC)], resource="/work/public.md")
        assert other.violation

    def test_decay_and_refresh(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        t.record(s, "fs", "read_file", "/a", [DOC], [], [])
        clock.advance(29 * 60)
        assert t.tainted(s)
        clock.advance(2 * 60)
        assert not t.tainted(s), "a record stops counting below weight 1/4 (30 min)"
        t.record(s, "fs", "read_file", "/a", [DOC], [], [])
        clock.advance(20 * 60)
        t.record(s, "fs", "read_file", "/a", [DOC], [], [])  # re-reading refreshes
        clock.advance(20 * 60)
        assert t.tainted(s)

    def test_allowed_flows(self, clock):
        t = TaintEngine(SecurityConfig(allow_flows=[("filesystem", "github")]), None, clock)
        s = self.session(t)
        t.record(s, "filesystem", "read_file", "/a", [DOC], [], [])
        d = t.check_call(s, "github", "write", False, [("body", SECRET)])
        assert not d.violation and d.allowed_flows
        granted = t.check_call(s, "mail", "egress", False, [("body", SECRET)], flow_allowed=lambda a, b: b == "mail")
        assert not granted.violation

    def test_embedding_near_paraphrase(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        emb = np.array([1.0, 0.2, 0.0, 0.0])
        t.record(s, "fs", "read_file", "/a", ["completely different words here"], [], [emb])
        d = t.check_call(s, "mail", "egress", False, [("body", "zzz")], [("body", np.array([0.98, 0.25, 0.0, 0.01]))])
        assert d.violation and d.overlaps[0].how == "embedding"

    def test_blob_hash(self, clock):
        import hashlib

        t = self.engine(clock)
        s = self.session(t)
        blob = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
        t.record(s, "fs", "read_image", None, [], [hashlib.sha256(blob.encode()).hexdigest()], [])
        assert t.check_call(s, "up", "egress", False, [("data", blob)]).violation

    def test_restore_after_restart_is_conservative(self, clock, store):
        from mcp_shield.winnow import fingerprint

        t = self.engine(clock, store)
        s = self.session(t)
        rec = t.record(s, "fs", "read_file", "/a", [DOC], [], [np.ones(4)])
        canonical, fps = fingerprint(DOC)
        p = int(fps.positions[0])
        short = "qq# " + canonical[p : p + 12] + " #qq"  # shares one kept 12-gram, far less than 32 chars
        assert rec is not None
        assert not t.check_call(s, "mail", "egress", False, [("body", short)]).violation, (
            "with the record text in memory the run is measured exactly"
        )
        t.escalate(s, "test")
        t2 = self.engine(clock, store)
        s2 = t2.sessions[s.key]
        assert s2.strict and s2.strict_reasons == ["test"]
        assert t2.tainted(s2)
        assert t2.check_call(s2, "mail", "egress", False, [("body", SECRET)]).violation
        assert t2.check_call(s2, "mail", "egress", False, [("body", short)]).violation, (
            "without the record text any fingerprint hit counts"
        )

    def test_needs_hold(self, clock):
        t = self.engine(clock)
        s = self.session(t)
        assert t.needs_hold(s, "execute", shell_mode=True, annotated=False) is None, "untainted session"
        t.record(s, "web", "fetch", None, ["some page"], [], [])
        assert t.needs_hold(s, "execute", shell_mode=True, annotated=False)
        assert t.needs_hold(s, "egress", shell_mode=False, annotated=True)
        assert t.needs_hold(s, "egress", shell_mode=False, annotated=False) is None
        t.escalate(s, "abstain")
        assert t.needs_hold(s, "egress", shell_mode=False, annotated=False)
        assert t.needs_hold(s, "write", shell_mode=False, annotated=False) is None

    def test_args_hash_exact(self):
        assert args_hash("s", "t", {"a": 1, "b": 2}) == args_hash("s", "t", {"b": 2, "a": 1})
        assert args_hash("s", "t", {"a": 1}) != args_hash("s", "t", {"a": 2})


class TestPinning:
    def test_canonical_hash_ignores_key_order(self):
        a = {"name": "x", "description": "d", "inputSchema": {"a": 1, "b": 2}}
        b = {"inputSchema": {"b": 2, "a": 1}, "description": "d", "name": "x", "noise": 1}
        assert def_hash(canonical_def("tool", a)) == def_hash(canonical_def("tool", b))

    def test_exposure_matrix(self, store):
        store.upsert_pin("s", "tool", "read_x", "h", {}, "read", ("allow", "benign", 0, 0))
        store.upsert_pin("s", "tool", "run_x", "h", {}, "execute", ("allow", "benign", 0, 0))
        store.upsert_pin("s", "tool", "bad", "h", {}, "read", ("block", "description_poisoning", 0.99, 0))
        store.upsert_pin("s", "tool", "noscan", "h", {}, "read", ("error", "unavailable", 0, 0))
        pins = store.pins("s")
        q = STATUS_QUARANTINED
        assert exposure(q, "tool", pins[("tool", "read_x")]).expose
        assert not exposure(q, "tool", pins[("tool", "run_x")]).expose
        assert exposure(q, "tool", pins[("tool", "bad")]).event == "injection"
        assert exposure(q, "tool", pins[("tool", "noscan")]).event == "scan_error"
        assert not exposure(q, "tool", None).expose
        store.accept_pins("s")
        pins = store.pins("s")
        assert exposure(STATUS_ACCEPTED, "tool", pins[("tool", "run_x")]).expose
        assert not exposure(STATUS_PROBE_QUARANTINED, "tool", pins[("tool", "run_x")]).expose
        store.upsert_pin("s", "tool", "new_tool", "h", {}, "read", ("allow", "benign", 0, 0))
        assert exposure(STATUS_ACCEPTED, "tool", store.pins("s")[("tool", "new_tool")]).event == "tamper"

    def test_known_manifest_and_diff(self):
        assert matches_known_manifest({"a", "b"}, {"ref": {"a", "b", "c"}}) == "ref"
        assert matches_known_manifest({"a", "z"}, {"ref": {"a", "b"}}) is None
        assert diff_definitions({"description": "x"}, {"description": "y"}) == [
            {"field": "description", "old": "x", "new": "y"}
        ]
