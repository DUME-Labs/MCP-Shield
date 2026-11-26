"""The localhost dashboard, served by the engine (07 §3).

A monitoring surface for a security product is itself an attack surface (CVE-2025-49596),
so: 127.0.0.1 only; a bearer token required on **every** endpoint, exchanged once for an
``HttpOnly`` ``SameSite=Strict`` cookie; ``Host`` and ``Origin`` validated on every
request; no CORS; mutating calls additionally need a custom header (a cross-site form
cannot send one); a strict Content-Security-Policy; and no endpoint can start a process.
Untrusted segments are rendered with ``textContent`` only.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import secrets
from importlib import resources
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .. import __version__

if TYPE_CHECKING:
    from ..engine.server import Engine

COOKIE = "mcp_shield_session"
CSRF_HEADER = "x-mcp-shield"
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml"}


def _static(name: str) -> bytes:
    return resources.files("mcp_shield.dashboard").joinpath("static").joinpath(name).read_bytes()


def make_dashboard_app(engine: Engine, token: str, port: int) -> Starlette:
    inspector = engine.inspector
    assert inspector is not None
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    sessions: set[str] = set()
    csp = (
        f"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        f"connect-src 'self' ws://127.0.0.1:{port} ws://localhost:{port}; base-uri 'none'; form-action 'none'; "
        f"frame-ancestors 'none'"
    )
    security_headers = {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "X-Frame-Options": "DENY",
    }

    def authed(conn: Request | WebSocket) -> bool:
        cookie = conn.cookies.get(COOKIE)
        if cookie and any(hmac.compare_digest(cookie, s) for s in sessions):
            return True
        auth = conn.headers.get("authorization", "")
        return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), token)

    def refuse(request: Request, *, mutating: bool = False) -> Response | None:
        if request.headers.get("host", "").lower() not in hosts:
            return Response("Host not allowed", status_code=403)
        origin = request.headers.get("origin")
        if origin is not None and origin not in origins:
            return Response("Origin not allowed", status_code=403)
        if not authed(request):
            return JSONResponse(
                {"error": "unauthorized — open the link `mcp-shield dashboard` prints"}, status_code=401
            )
        if mutating and request.headers.get(CSRF_HEADER) != "1":
            return Response("Missing anti-CSRF header", status_code=403)
        return None

    def ok(data: Any, status: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status, headers=security_headers)

    async def index(request: Request) -> Response:
        if request.headers.get("host", "").lower() not in hosts:
            return Response("Host not allowed", status_code=403)
        given = request.query_params.get("token")
        if given is not None:
            if not hmac.compare_digest(given, token):
                return Response("Invalid token", status_code=401)
            sid = secrets.token_urlsafe(32)
            sessions.add(sid)
            resp = RedirectResponse("/", status_code=303, headers=security_headers)
            resp.set_cookie(COOKIE, sid, httponly=True, samesite="strict", secure=False, path="/")
            return resp
        if not authed(request):
            return Response(
                "MCP-Shield dashboard: open the one-time link printed by `mcp-shield dashboard`.",
                status_code=401,
                media_type="text/plain",
                headers=security_headers,
            )
        return Response(_static("index.html"), media_type="text/html", headers=security_headers)

    async def static(request: Request) -> Response:
        refused = refuse(request)
        if refused:
            return refused
        name = request.path_params["name"]
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if "/" in name or "\\" in name or suffix not in STATIC_TYPES or name == "index.html":
            return Response(status_code=404)
        try:
            body = _static(name)
        except (FileNotFoundError, OSError):
            return Response(status_code=404)
        return Response(body, media_type=STATIC_TYPES[suffix], headers=security_headers)

    async def status(request: Request) -> Response:
        return refuse(request) or ok(engine.status() | {"dashboard_version": __version__})

    async def events(request: Request) -> Response:
        refused = refuse(request)
        if refused:
            return refused
        q = request.query_params
        verdicts = [v for v in q.get("verdict", "").split(",") if v] or None
        rows = inspector.store.events(
            min(int(q.get("limit", "200")), 1000),
            verdicts=verdicts,
            server=q.get("server") or None,
            query=q.get("q") or None,
        )
        for row in rows:
            row["has_segment"] = bool(row.pop("segment", None))
        return ok(rows)

    async def event(request: Request) -> Response:
        refused = refuse(request)
        if refused:
            return refused
        row = inspector.store.event(request.path_params["ref"])
        return ok(row) if row else ok({"error": "not found"}, 404)

    async def event_action(request: Request) -> Response:
        refused = refuse(request, mutating=True)
        if refused:
            return refused
        body = await request.json()
        try:
            result = inspector.action(request.path_params["ref"], str(body.get("action")), body.get("params") or {})
        except (KeyError, ValueError) as exc:
            return ok({"error": str(exc)}, 400)
        return ok(result)

    async def servers(request: Request) -> Response:
        return refuse(request) or ok(inspector.store.servers())

    async def pins(request: Request) -> Response:
        refused = refuse(request)
        if refused:
            return refused
        server = request.path_params["server"]
        return ok({"server": inspector.store.server(server), "items": inspector.pins_view(server)})

    async def accept(request: Request) -> Response:
        refused = refuse(request, mutating=True)
        if refused:
            return refused
        body = await request.json()
        if body.get("confirm") is not True:
            return ok({"error": "re-pinning requires explicit confirmation"}, 400)
        items = [tuple(i) for i in body.get("items") or []] or None
        n = inspector.accept_pins(request.path_params["server"], items)  # type: ignore[arg-type]
        return ok({"accepted": n})

    async def release(request: Request) -> Response:
        refused = refuse(request, mutating=True)
        if refused:
            return refused
        inspector.release_server(request.path_params["server"])
        return ok({"ok": True})

    async def sessions_view(request: Request) -> Response:
        return refuse(request) or ok(inspector.sessions_view())

    async def reset(request: Request) -> Response:
        refused = refuse(request, mutating=True)
        if refused:
            return refused
        return ok({"ok": inspector.reset_session(request.path_params["key"])})

    async def review(request: Request) -> Response:
        refused = refuse(request)
        if refused:
            return refused
        rows = inspector.store.review_queue()
        return ok(
            {
                "count": len(rows),
                "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in {r["kind"] for r in rows}},
            }
        )

    async def feed(ws: WebSocket) -> None:
        if ws.headers.get("host", "").lower() not in hosts or ws.headers.get("origin") not in origins or not authed(ws):
            await ws.close(code=4403)
            return
        await ws.accept()
        q = inspector.bus.subscribe()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), 25)
                except asyncio.TimeoutError:
                    item = {"type": "ping"}
                await ws.send_json(item)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            inspector.bus.unsubscribe(q)
            with contextlib.suppress(Exception):
                await ws.close()

    routes = [
        Route("/", index),
        Route("/static/{name}", static),
        Route("/api/status", status),
        Route("/api/events", events),
        Route("/api/events/{ref}", event),
        Route("/api/events/{ref}/action", event_action, methods=["POST"]),
        Route("/api/servers", servers),
        Route("/api/pins/{server}", pins),
        Route("/api/pins/{server}/accept", accept, methods=["POST"]),
        Route("/api/servers/{server}/release", release, methods=["POST"]),
        Route("/api/sessions", sessions_view),
        Route("/api/sessions/{key}/reset", reset, methods=["POST"]),
        Route("/api/review", review),
        WebSocketRoute("/ws", feed),
    ]
    return Starlette(routes=routes)
