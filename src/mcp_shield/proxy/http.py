"""Streamable HTTP listener (02 §1) — an ASGI reverse proxy that runs inside the engine.

* Binds to 127.0.0.1; validates ``Host`` against its own address, rejects any ``Origin``
  not on the allowlist (empty by default — native hosts send none) and, when configured,
  requires its own bearer token.
* Tier 1's header allowlist runs on ``Mcp-Method`` / ``Mcp-Name`` **before the body is
  read**; header ↔ body consistency is checked once it is (``-32020``, HTTP 400).
* Transparent on the server side: the host's headers and ``_meta`` are forwarded as-is
  and nothing identifying is added (no ``Via``, ``X-Forwarded-*``, ``User-Agent`` or
  ``Server``). JSON and SSE responses are inspected message by message.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..cascade.argpolicy import RULES_BOTH
from ..cascade.tier1_policy import header_precheck
from ..config import ListenerConfig, SecurityConfig
from ..engine.core import Connection, Inspector
from ..errors import stub_error
from ..jsonrpc import dumps, error_response, make_error
from ..protocol import HDR_SESSION_LEGACY, PARSE_ERROR

MAX_REQUEST_BODY = 16 * 1024 * 1024
MAX_RESPONSE_BODY = 64 * 1024 * 1024
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)
_RESPONSE_DROP = _HOP_BY_HOP | {"content-encoding", "server", "date"}


class _BodyTooLarge(Exception):
    pass


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise _BodyTooLarge
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _sse_events(lines: list[str]) -> dict[str, str]:
    event: dict[str, str] = {}
    data: list[str] = []
    for line in lines:
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data.append(value)
        elif field in ("event", "id", "retry"):
            event[field] = value
    if data:
        event["data"] = "\n".join(data)
    return event


def _format_sse(event: dict[str, str], data: str | None) -> str:
    out = []
    for key in ("id", "event", "retry"):
        if key in event:
            out.append(f"{key}: {event[key]}")
    if data is not None:
        out.extend(f"data: {line}" for line in data.split("\n"))
    return "\n".join(out) + "\n\n"


def make_listener_app(listener: ListenerConfig, inspector: Inspector, config: SecurityConfig) -> Starlette:
    upstream = listener.upstream
    upstream_host = urlsplit(upstream).hostname or ""
    local_upstream = upstream_host in ("127.0.0.1", "localhost", "::1")
    port = listener.port
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    origins = set(config.origin_allowlist)
    server_policy = config.servers.get(listener.name)
    remote = True if server_policy is None or server_policy.remote is None else server_policy.remote
    legacy_clients: dict[str, dict[str, Any]] = {}
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=None, write=60.0), follow_redirects=False, trust_env=False
    )
    client.headers = httpx.Headers()  # no default User-Agent / Accept-Encoding: stay transparent

    def gate(request: Request) -> Response | None:
        if request.headers.get("host", "").lower() not in allowed_hosts:
            return Response("Host not allowed", status_code=403)
        origin = request.headers.get("origin")
        if origin is not None and origin not in origins:
            return Response("Origin not allowed", status_code=403)
        if listener.bearer_token:
            expected = f"Bearer {listener.bearer_token}"
            if request.headers.get("authorization", "") != expected:
                return Response("Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return None

    def connection(request: Request) -> Connection:
        auth = request.headers.get("authorization", "")
        if listener.bearer_token:
            subject = "listener-token"
        elif auth.lower().startswith("bearer "):
            subject = "bearer:" + hashlib.sha256(auth[7:].strip().encode()).hexdigest()[:16]
        else:
            subject = None
        conn = Connection(
            listener.name,
            "http",
            remote,
            f"http:{request.client.host if request.client else '?'}",
            RULES_BOTH,
            local_upstream,
            subject=subject,
        )
        sid = request.headers.get(HDR_SESSION_LEGACY)
        if sid and sid in legacy_clients:
            conn.client_info = legacy_clients[sid]
        return conn

    def upstream_headers(request: Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        if listener.bearer_token:
            headers.pop("authorization", None)
            headers.pop("Authorization", None)
        return headers

    def response_headers(resp: httpx.Response) -> dict[str, str]:
        return {k: v for k, v in resp.headers.items() if k.lower() not in _RESPONSE_DROP}

    async def answer_server(request: Request, messages: list[dict[str, Any]]) -> None:
        """Send the proxy's own replies (declines) to the server, as a host would."""
        for msg in messages:
            headers = upstream_headers(request)
            headers["content-type"] = "application/json"
            headers.pop("mcp-method", None)
            headers.pop("mcp-name", None)
            with contextlib.suppress(httpx.HTTPError):
                await client.post(upstream, content=dumps(msg).encode("utf-8"), headers=headers)

    async def relay_response(request: Request, conn: Connection, resp: httpx.Response) -> Response:
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        sid = resp.headers.get(HDR_SESSION_LEGACY)
        if sid and conn.client_info:
            legacy_clients[sid] = conn.client_info
        if ctype == "text/event-stream":
            return StreamingResponse(
                stream_events(request, conn, resp),
                status_code=resp.status_code,
                headers=response_headers(resp),
                media_type="text/event-stream",
            )
        try:
            body = await resp.aread()
        finally:
            await resp.aclose()
        if len(body) > MAX_RESPONSE_BODY:
            ref = inspector.log.emit(
                "dos",
                "block",
                tier="tier2",
                server=listener.name,
                direction="s2h",
                reason=f"HTTP response of {len(body)} bytes exceeds {MAX_RESPONSE_BODY}",
            )
            return JSONResponse(error_response(None, stub_error(ref)), status_code=200)
        if not body:
            return Response(status_code=resp.status_code, headers=response_headers(resp))
        try:
            message = json.loads(body)
        except ValueError:
            message = None
        if message is None or not isinstance(message, dict):
            # not JSON-RPC: never hand unknown server text to the host; keep status and auth challenge only
            keep = {k: v for k, v in resp.headers.items() if k.lower() in ("www-authenticate", HDR_SESSION_LEGACY)}
            return Response(status_code=resp.status_code if resp.status_code >= 400 else 502, headers=keep)
        decision = await inspector.from_server(conn, message)
        if decision.to_server:
            await answer_server(request, decision.to_server)
        headers = response_headers(resp)
        headers.pop("content-type", None)
        if not decision.to_host:
            return Response(status_code=202, headers=headers)
        return Response(
            dumps(decision.to_host[0]), status_code=resp.status_code, headers=headers, media_type="application/json"
        )

    async def stream_events(request: Request, conn: Connection, resp: httpx.Response) -> AsyncIterator[str]:
        try:
            lines: list[str] = []
            async for line in resp.aiter_lines():
                if line != "":
                    lines.append(line)
                    continue
                if not lines:
                    continue
                event = _sse_events(lines)
                lines = []
                data = event.pop("data", None)
                if data is None:
                    yield _format_sse(event, None)
                    continue
                try:
                    message = json.loads(data)
                except ValueError:
                    continue  # never forward non-JSON server text
                decision = await inspector.from_server(conn, message)
                if decision.to_server:
                    await answer_server(request, decision.to_server)
                for out in decision.to_host:
                    yield _format_sse(event, dumps(out))
        finally:
            await resp.aclose()

    async def post(request: Request) -> Response:
        conn = connection(request)
        reason = header_precheck(dict(request.headers), lambda m, n: inspector.name_allowed(listener.name, m, n))
        if reason:
            ref = inspector.log.emit(
                "policy",
                "block",
                tier="tier1",
                server=listener.name,
                direction="h2s",
                reason=f"header allowlist (before body): {reason}",
            )
            return JSONResponse(error_response(None, stub_error(ref)), status_code=403)
        try:
            body = await _read_body(request, MAX_REQUEST_BODY)
        except _BodyTooLarge:
            ref = inspector.log.emit(
                "policy", "block", tier="tier1", server=listener.name, direction="h2s", reason="request body too large"
            )
            return JSONResponse(error_response(None, stub_error(ref)), status_code=413)
        try:
            message = json.loads(body)
        except ValueError:
            return JSONResponse(error_response(None, make_error(PARSE_ERROR, "Parse error")), status_code=400)
        decision = await inspector.from_host(conn, message, headers=dict(request.headers))
        if decision.http_status:
            body_out = (
                decision.to_host[0] if decision.to_host else error_response(None, make_error(-32020, "Header mismatch"))
            )
            return JSONResponse(body_out, status_code=decision.http_status)
        if not decision.to_server:
            if decision.to_host:
                return JSONResponse(decision.to_host[0], status_code=200)
            return Response(status_code=202)
        outgoing = decision.to_server[0]
        try:
            req = client.build_request(
                "POST",
                upstream,
                content=dumps(outgoing).encode("utf-8"),
                headers=upstream_headers(request),
                params=request.query_params or None,
            )
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            inspector.log.emit("upstream", "info", server=listener.name, reason=f"upstream unreachable: {exc}")
            return Response("Upstream unavailable", status_code=502)
        return await relay_response(request, conn, resp)

    async def get(request: Request) -> Response:
        conn = connection(request)
        try:
            req = client.build_request(
                "GET", upstream, headers=upstream_headers(request), params=request.query_params or None
            )
            resp = await client.send(req, stream=True)
        except httpx.HTTPError:
            return Response("Upstream unavailable", status_code=502)
        return await relay_response(request, conn, resp)

    async def delete(request: Request) -> Response:
        try:
            resp = await client.request("DELETE", upstream, headers=upstream_headers(request))
        except httpx.HTTPError:
            return Response("Upstream unavailable", status_code=502)
        return Response(
            status_code=resp.status_code,
            headers={k: v for k, v in response_headers(resp).items() if k.lower() != "content-type"},
        )

    async def endpoint(request: Request) -> Response:
        refused = gate(request)
        if refused is not None:
            inspector.log.emit(
                "listener",
                "block",
                server=listener.name,
                tier="tier1",
                reason=f"HTTP request refused ({refused.status_code}): host/origin/token check",
            )
            return refused
        if request.method == "POST":
            return await post(request)
        if request.method == "GET":
            return await get(request)
        if request.method == "DELETE":
            return await delete(request)
        return Response(status_code=405)

    app = Starlette(routes=[Route("/{path:path}", endpoint, methods=["GET", "POST", "DELETE"])])
    app.state.upstream_client = client
    return app
