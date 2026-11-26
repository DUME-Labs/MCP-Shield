"""Dashboard: token → cookie, Host/Origin checks, CSRF header, CSP, actions, WebSocket origin."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mcp_shield.config import SecurityConfig
from mcp_shield.dashboard.app import COOKIE, make_dashboard_app
from mcp_shield.engine.core import Inspector
from mcp_shield.engine.server import Engine

PORT = 8599
BASE = f"http://127.0.0.1:{PORT}"
TOKEN = "t" * 43


@pytest.fixture
def setup(classifier, store):
    engine = Engine(SecurityConfig(), dashboard=False)
    engine.inspector = Inspector(SecurityConfig(), store, classifier)
    engine.started = time.time()
    app = make_dashboard_app(engine, TOKEN, PORT)
    client = TestClient(app, base_url=BASE)
    yield engine, client
    engine.inspector.close()


def login(client: TestClient) -> None:
    r = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")


def test_every_endpoint_needs_the_token(setup):
    _, client = setup
    for path in ("/", "/api/status", "/api/events", "/api/sessions", "/static/app.js"):
        assert client.get(path).status_code == 401, path
    assert client.get("/?token=wrong").status_code == 401


def test_login_then_browse(setup):
    _engine, client = setup
    login(client)
    r = client.get("/")
    assert r.status_code == 200 and "MCP-Shield" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    assert client.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/static/../server.py").status_code == 404
    assert client.get("/api/status").json()["classifier"]["available"]


def test_bearer_header_works_for_scripts(setup):
    _, client = setup
    assert client.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_host_and_origin_checked(setup):
    _, client = setup
    login(client)
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/status", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/status", headers={"Origin": BASE}).status_code == 200


def test_mutations_need_csrf_header_and_confirmation(setup):
    engine, client = setup
    engine.inspector.store.touch_server("fs", "stdio", False)
    engine.inspector.store.upsert_pin(
        "fs", "tool", "write_file", "h", {"name": "write_file"}, "write", ("allow", "benign", 0.0, 0.0)
    )
    login(client)
    assert client.post("/api/pins/fs/accept", json={"confirm": True}).status_code == 403
    hdr = {"X-MCP-Shield": "1"}
    assert client.post("/api/pins/fs/accept", json={}, headers=hdr).status_code == 400
    r = client.post("/api/pins/fs/accept", json={"confirm": True}, headers=hdr)
    assert r.json() == {"accepted": 1}
    assert client.get("/api/pins/fs").json()["items"][0]["accepted"]


def test_event_actions(setup):
    engine, client = setup
    ref = engine.inspector.log.emit(
        "taint",
        "block",
        server="mail",
        tool="send_email",
        session="s1",
        args_hash="abc",
        sources=["fs"],
        segment="AKIAIOSFODNN7EXAMPLE secret",
    )
    login(client)
    event = client.get(f"/api/events/{ref}").json()
    assert "AKIAIOSFODNN7EXAMPLE" not in event["segment"], "segments are stored redacted"
    listed = client.get("/api/events?verdict=block").json()
    assert listed[0]["ref"] == ref and "segment" not in listed[0] and listed[0]["has_segment"]
    hdr = {"X-MCP-Shield": "1"}
    r = client.post(f"/api/events/{ref}/action", json={"action": "allow_once"}, headers=hdr)
    assert "single-use grant" in r.json()["note"]
    assert client.post(f"/api/events/{ref}/action", json={"action": "raise_cap"}, headers=hdr).status_code == 400
    assert client.post("/api/events/nope/action", json={"action": "allow_once"}, headers=hdr).status_code == 400


def test_websocket_origin_and_feed(setup):
    engine, client = setup
    login(client)
    # the TestClient opens WebSockets as ws://testserver/…: send the Host and session cookie a browser would send
    host = {"Host": f"127.0.0.1:{PORT}", "Cookie": f"{COOKIE}={client.cookies.get(COOKIE)}"}
    bad_origin = {**host, "Origin": "https://evil.example"}
    bad_host = {"Origin": BASE, "Cookie": host["Cookie"]}
    for headers in (bad_origin, bad_host):
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers=headers) as ws:
            ws.receive_json()
    with client.websocket_connect("/ws", headers={**host, "Origin": BASE}) as ws:
        engine.inspector.log.emit("policy", "block", server="x", reason="test")
        msg = ws.receive_json()
        assert msg["type"] == "event" and msg["event"]["kind"] == "policy" and "segment" not in msg["event"]


def test_no_endpoint_starts_a_process(setup):
    _, client = setup
    routes = {r.path for r in client.app.routes}
    assert not any(word in path for path in routes for word in ("spawn", "exec", "run", "install", "command"))
    assert COOKIE == "mcp_shield_session"
