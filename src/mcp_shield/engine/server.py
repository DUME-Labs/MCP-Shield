"""The engine — one process per OS user (02 §1, process model).

It holds the classifier, the Tier 2 cache, the session taint store, the pin store and
the event store; serves the local IPC channel the front-ends and the CLI talk to; runs
the HTTP listeners; and serves the dashboard. It never spawns an MCP server: no IPC
message, network request or dashboard action can start a process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import signal
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from .. import __version__
from ..cascade.argpolicy import RULES_BOTH, host_rules
from ..cascade.tier3_classifier import Classifier
from ..config import ListenerConfig, SecurityConfig
from ..paths import data_dir, ensure_private_dir, write_private_file
from ..refid import new_ref
from ..store import Store
from . import ipc
from .core import Connection, Inspector, fail_closed

log = logging.getLogger(__name__)

STATE_FILE = "engine.json"
TOKEN_FILE = "dashboard.token"  # noqa: S105 - a file name, not a secret
MAINTENANCE_INTERVAL_S = 60.0
PURGE_INTERVAL_S = 600.0


def dashboard_token(create: bool = True) -> str:
    path = data_dir() / TOKEN_FILE
    if path.exists():
        token = path.read_text(encoding="ascii").strip()
        if len(token) >= 32:
            return token
    if not create:
        raise FileNotFoundError(path)
    token = secrets.token_urlsafe(32)
    write_private_file(path, token.encode("ascii"))
    return token


def read_state() -> dict[str, Any] | None:
    path = data_dir() / STATE_FILE
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


try:
    import uvicorn

    class QuietServer(uvicorn.Server):
        """A uvicorn server that leaves signal handling to the engine (several run in one loop)."""

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

        def install_signal_handlers(self) -> None:  # uvicorn < 0.29
            return None
except ImportError:  # pragma: no cover
    QuietServer = None  # type: ignore[assignment,misc]


class Engine:
    def __init__(
        self,
        config: SecurityConfig,
        *,
        foreground: bool = False,
        classifier: Classifier | None = None,
        store: Store | None = None,
        idle_exit_s: float = 600.0,
        dashboard: bool = True,
        address: str | None = None,
    ) -> None:
        self.config = config
        self.foreground = foreground
        self.idle_exit_s = idle_exit_s
        self.with_dashboard = dashboard
        self.address = address or ipc.endpoint()
        self._classifier = classifier
        self._store = store
        self.inspector: Inspector | None = None
        self.frontends: dict[int, dict[str, Any]] = {}
        self.http_servers: dict[str, tuple[Any, asyncio.Task[None], ListenerConfig]] = {}
        self.dashboard_server: tuple[Any, asyncio.Task[None]] | None = None
        self.dashboard_url: str | None = None
        self.started = 0.0
        self._last_activity = time.time()
        self._stop = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._ipc: ipc.Listener | None = None
        self._next_frontend = 0
        self._writers: set[asyncio.StreamWriter] = set()

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        ensure_private_dir(data_dir())
        self.secret = ipc.load_secret()
        store = self._store or Store()
        classifier = self._classifier
        if classifier is None:
            from ..models import load_configured_classifier

            classifier = await asyncio.get_running_loop().run_in_executor(None, load_configured_classifier, self.config)
        self.inspector = Inspector(self.config, store, classifier)
        reason = getattr(classifier, "reason", None)
        if reason:
            self.inspector.log.emit("unavailable", "info", reason=f"classifier unavailable — failing closed: {reason}")
        try:
            self._ipc = await ipc.start_server(self._handle, self.address)
        except BaseException:  # e.g. another engine already listens: release what this one opened
            self.inspector.close()
            self.inspector.store.close()
            raise
        self.started = time.time()
        for listener in self.config.listeners:
            try:
                await self.open_listener(listener)
            except Exception as exc:  # a busy port must not take the engine down
                self.inspector.log.emit(
                    "warning", "info", server=listener.name, reason=f"listener {listener.listen} could not start: {exc}"
                )
        if self.with_dashboard:
            try:
                await self.start_dashboard()
            except Exception as exc:
                self.inspector.log.emit("warning", "info", reason=f"dashboard could not start: {exc}")
        self._write_state()
        self._spawn(self._maintenance())
        for msg in self.config.warnings:
            self.inspector.log.emit("warning", "info", reason=f"config: {msg}")

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def serve_forever(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if sig is not None:
                with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.add_signal_handler(sig, self._stop.set)
        try:
            await self._stop.wait()
        finally:
            await self.stop()

    def request_stop(self) -> None:
        self._stop.set()

    async def stop(self) -> None:
        if self._ipc is not None:
            await self._ipc.aclose()
            self._ipc = None
        await asyncio.sleep(0)  # let handlers of just-accepted connections register their writers
        for writer in list(self._writers):
            writer.close()
        for writer in list(self._writers):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), 2)
        self._writers.clear()
        for name in list(self.http_servers):
            await self.close_listener(name)
        if self.dashboard_server is not None:
            server, task = self.dashboard_server
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, 5)
            self.dashboard_server = None
        for task in list(self._tasks):
            task.cancel()
        if self.inspector is not None:
            self.inspector.close()
            self.inspector.store.close()
        with contextlib.suppress(OSError):
            state = read_state()
            if state and state.get("pid") == os.getpid():
                (data_dir() / STATE_FILE).unlink()

    def _write_state(self) -> None:
        state = {
            "pid": os.getpid(),
            "version": __version__,
            "started": self.started,
            "endpoint": self.address,
            "dashboard": self.dashboard_url,
            "listeners": {n: c.listen for n, (_, _, c) in self.http_servers.items()},
        }
        write_private_file(data_dir() / STATE_FILE, json.dumps(state, indent=2).encode("utf-8"))

    async def _maintenance(self) -> None:
        last_purge = 0.0
        while True:
            await asyncio.sleep(MAINTENANCE_INTERVAL_S)
            assert self.inspector is not None
            now = time.time()
            self.inspector.taint.expire()
            if now - last_purge > PURGE_INTERVAL_S:
                last_purge = now
                self.inspector.store.purge(now, self.config.retention_days, self.inspector.taint.live_horizon)
            if self._idle(now):
                log.info("engine idle for %.0f s with nothing attached; exiting", self.idle_exit_s)
                self._stop.set()

    def _idle(self, now: float) -> bool:
        if self.foreground or self.frontends or self.http_servers:
            return False
        if self.inspector is not None and self.inspector.bus._subscribers:
            return False
        return now - self._last_activity > self.idle_exit_s

    # ------------------------------------------------------------------ IPC

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            await self._serve_connection(reader, writer)
        finally:
            self._writers.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), 2)

    async def _serve_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await ipc.server_handshake(reader, writer, self.secret)
        except (ipc.IPCError, asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError) as exc:
            if self.inspector is not None:
                self.inspector.log.emit("ipc", "block", reason=f"IPC client refused: {exc}")
            return
        lock = asyncio.Lock()
        conn: Connection | None = None
        fid: int | None = None
        pending: set[asyncio.Task[None]] = set()

        async def send(obj: dict[str, Any]) -> None:
            async with lock:
                await ipc.write_frame(writer, obj)

        try:
            while True:
                frame = await ipc.read_frame(reader)
                self._last_activity = time.time()
                kind = frame.get("t")
                if kind == "hello":
                    conn = Connection(
                        server=str(frame.get("server") or "unnamed"),
                        transport=str(frame.get("transport", "stdio")),
                        remote=bool(frame.get("remote", False)),
                        host_id=str(frame.get("host_id", "")),
                        rules=host_rules(),
                        local=True,
                    )
                    assert self.inspector is not None
                    self.inspector.register(conn, command=frame.get("command"))
                    self._next_frontend += 1
                    fid = self._next_frontend
                    self.frontends[fid] = {
                        "server": conn.server,
                        "host": conn.host_id,
                        "pid": frame.get("pid"),
                        "since": time.time(),
                        "command": frame.get("command"),
                    }
                    await send({"t": "hello_ok", "engine": __version__, "pid": os.getpid()})
                elif kind == "msg":
                    if conn is None:
                        await send({"t": "error", "error": "hello first"})
                        continue
                    task = asyncio.create_task(self._inspect(conn, frame, send))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                elif kind == "cmd":
                    try:
                        result = await self.command(str(frame.get("cmd")), frame.get("args") or {})
                        await send({"t": "cmd_result", "id": frame.get("id"), "ok": True, "result": result})
                    except Exception as exc:
                        await send({"t": "cmd_result", "id": frame.get("id"), "ok": False, "error": str(exc)})
                else:
                    await send({"t": "error", "error": f"unknown frame type {kind!r}"})
        except (asyncio.IncompleteReadError, ConnectionError, ipc.IPCError, ValueError):
            pass
        finally:
            for task in pending:
                task.cancel()
            if fid is not None:
                self.frontends.pop(fid, None)
            self._last_activity = time.time()

    async def _inspect(self, conn: Connection, frame: dict[str, Any], send: Any) -> None:
        assert self.inspector is not None
        direction = frame.get("dir")
        message = frame.get("message")
        try:
            if direction == "h2s":
                decision = await self.inspector.from_host(conn, message)
            else:
                decision = await self.inspector.from_server(conn, message)
            out = decision.to_json()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # an engine bug fails closed for this message, never open
            log.exception("inspection failed")
            ref = self.inspector.log.emit(
                "error",
                "block",
                server=conn.server,
                direction=str(direction),
                reason=f"internal error, failing closed: {type(exc).__name__}: {exc}",
            )
            out = fail_closed(str(direction), message, ref).to_json()
        with contextlib.suppress(ConnectionError, RuntimeError):
            await send({"t": "verdict", "seq": frame.get("seq"), "decision": out})

    # ------------------------------------------------------------------ commands (CLI + front-ends)

    async def command(self, cmd: str, args: dict[str, Any]) -> Any:
        insp = self.inspector
        assert insp is not None
        if cmd == "status":
            return self.status()
        if cmd == "servers":
            return insp.store.servers()
        if cmd == "pins":
            return {"server": insp.store.server(str(args["server"])), "items": insp.pins_view(str(args["server"]))}
        if cmd == "accept_pins":
            items = [tuple(i) for i in args.get("items") or []] or None
            return {"accepted": insp.accept_pins(str(args["server"]), items)}  # type: ignore[arg-type]
        if cmd == "release_server":
            insp.release_server(str(args["server"]))
            return {"ok": True}
        if cmd == "action":
            return insp.action(str(args["ref"]), str(args["action"]), args.get("params") or {})
        if cmd == "reset_session":
            return {"ok": insp.reset_session(str(args["key"]))}
        if cmd == "events":
            return insp.store.events(int(args.get("limit", 50)))
        if cmd == "event":
            return insp.store.event(str(args["ref"]))
        if cmd == "open_listener":
            listener = ListenerConfig(
                name=str(args["name"]),
                upstream=str(args["upstream"]),
                listen=str(args.get("listen", "127.0.0.1:8800")),
                bearer_token=args.get("bearer_token"),
            )
            await self.open_listener(listener)
            return {"listen": listener.listen}
        if cmd == "dashboard_url":
            if self.dashboard_server is None:
                await self.start_dashboard(int(args.get("port") or self.config.dashboard_port))
            return {"url": f"{self.dashboard_url}/?token={dashboard_token()}"}
        if cmd == "shutdown":
            asyncio.get_running_loop().call_later(0.1, self._stop.set)
            return {"ok": True}
        raise ValueError(f"unknown command {cmd!r}")

    def status(self) -> dict[str, Any]:
        insp = self.inspector
        assert insp is not None
        clf = insp.classifier
        return {
            "pid": os.getpid(),
            "version": __version__,
            "uptime_s": round(time.time() - self.started, 1),
            "foreground": self.foreground,
            "endpoint": self.address,
            "classifier": {
                "name": clf.name,
                "version": clf.version,
                "available": not hasattr(clf, "reason"),
                "reason": getattr(clf, "reason", None),
                "providers": getattr(clf, "providers", None),
                "model_file": getattr(clf, "model_file", None),
            },
            "fail_open": self.config.fail_open,
            "taint_policy": self.config.taint_policy,
            "frontends": list(self.frontends.values()),
            "listeners": {
                n: asdict(c) | {"bearer_token": bool(c.bearer_token)} for n, (_, _, c) in self.http_servers.items()
            },
            "dashboard": self.dashboard_url,
            "stats": dict(insp.stats),
            "sessions": len(insp.taint.sessions),
            "live_taint_records": len(insp.taint.records),
        }

    # ------------------------------------------------------------------ HTTP listeners + dashboard

    async def _serve_uvicorn(self, app: Any, host: str, port: int) -> tuple[Any, asyncio.Task[None]]:
        import uvicorn

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            lifespan="off",
            server_header=False,
            date_header=False,
            access_log=False,
            ws_max_size=1 << 20,
        )
        server = QuietServer(config)
        task = asyncio.create_task(server.serve())
        for _ in range(200):
            if server.started:
                return server, task
            if task.done():
                exc = task.exception() if not task.cancelled() else None
                raise OSError(f"could not listen on {host}:{port}: {exc or 'server exited'}")
            await asyncio.sleep(0.025)
        server.should_exit = True
        raise OSError(f"timed out starting a listener on {host}:{port}")

    async def open_listener(self, listener: ListenerConfig) -> None:
        from ..proxy.http import make_listener_app

        if listener.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("listeners bind to 127.0.0.1 only")
        if listener.name in self.http_servers:
            existing = self.http_servers[listener.name][2]
            if existing.listen == listener.listen and existing.upstream == listener.upstream:
                return
            await self.close_listener(listener.name)
        assert self.inspector is not None
        app = make_listener_app(listener, self.inspector, self.config)
        server, task = await self._serve_uvicorn(app, listener.host, listener.port)
        self.http_servers[listener.name] = (server, task, listener)
        self.inspector.register(Connection(listener.name, "http", True, "", RULES_BOTH, False))
        self._write_state()

    async def close_listener(self, name: str) -> None:
        entry = self.http_servers.pop(name, None)
        if entry is None:
            return
        server, task, _ = entry
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 5)
        # the listener's pooled connections to its upstream server close with it
        upstream = getattr(getattr(server.config.app, "state", None), "upstream_client", None)
        if upstream is not None:
            with contextlib.suppress(Exception):
                await upstream.aclose()

    async def start_dashboard(self, port: int | None = None) -> None:
        from ..dashboard.app import make_dashboard_app

        assert self.inspector is not None
        port = port or self.config.dashboard_port
        app = make_dashboard_app(self, dashboard_token(), port)
        self.dashboard_server = await self._serve_uvicorn(app, "127.0.0.1", port)
        self.dashboard_url = f"http://127.0.0.1:{port}"
        self._write_state()


# --------------------------------------------------------------------------- entry point


def new_event_loop() -> asyncio.AbstractEventLoop:
    """Proactor on Windows (named pipes, subprocess pipes); uvloop on POSIX when installed."""
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    try:
        import uvloop

        return uvloop.new_event_loop()  # type: ignore[no-any-return]
    except ImportError:
        return asyncio.new_event_loop()


def run_engine(config: SecurityConfig, *, foreground: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s mcp-shield engine %(levelname)s %(message)s")

    async def main() -> int:
        engine = Engine(config, foreground=foreground)
        try:
            await engine.start()
        except ipc.IPCError as exc:
            log.error("%s", exc)
            return 3
        log.info("engine %s ready on %s (pid %d)", __version__, engine.address, os.getpid())
        await engine.serve_forever()
        return 0

    loop = new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()


def new_ref_local() -> str:
    return new_ref()
