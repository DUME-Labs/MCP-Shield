"""The stdio front-end: ``mcp-shield proxy --name <server> -- <command…>`` (02 §1).

The host launches this process instead of the server. It spawns **only** the command on
its own command line, relays every newline-delimited JSON-RPC message to the engine and
writes back whatever the verdict says. Messages are inspected concurrently but released
in arrival order per direction. If the engine cannot be reached or started, traffic is
blocked in both directions (fail closed) while the front-end keeps trying to reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import shlex
import shutil
import signal
import sys
import threading
from collections.abc import Callable
from typing import Any

from ..engine.client import EngineClient, EngineUnavailable
from ..engine.core import fail_closed
from ..jsonrpc import dumps, error_response, make_error
from ..protocol import PARSE_ERROR
from ..refid import new_ref
from .hostid import host_identity

STREAM_LIMIT = 64 * 1024 * 1024


def _err(text: str) -> None:
    sys.stderr.write(f"[mcp-shield] {text}\n")
    sys.stderr.flush()


def resolve_command(command: list[str]) -> list[str]:
    """Resolve ``npx``-style launchers through PATH (and PATHEXT on Windows)."""
    if not command:
        raise ValueError("no server command given after --")
    exe = shutil.which(command[0])
    return [exe or command[0], *command[1:]]


class _HostWriter(threading.Thread):
    """Writes to the host's stdout from a thread, so a slow host never blocks the event loop."""

    def __init__(self, stream: Any) -> None:
        super().__init__(name="mcp-shield-host-writer", daemon=True)
        self.stream = stream
        self.q: queue.Queue[bytes | None] = queue.Queue()

    def run(self) -> None:
        while True:
            data = self.q.get()
            if data is None:
                return
            try:
                self.stream.write(data)
                self.stream.flush()
            except (OSError, ValueError):
                return

    def send(self, message: dict[str, Any]) -> None:
        self.q.put(dumps(message).encode("utf-8") + b"\n")


def _start_host_reader(loop: asyncio.AbstractEventLoop, stream: Any, out: asyncio.Queue[bytes | None]) -> None:
    """Read the host's stdin in a thread (anonymous pipes are not overlapped on Windows)."""

    def run() -> None:
        try:
            for line in iter(stream.readline, b""):
                loop.call_soon_threadsafe(out.put_nowait, line)
        except (OSError, ValueError):
            pass
        finally:
            loop.call_soon_threadsafe(out.put_nowait, None)

    threading.Thread(target=run, name="mcp-shield-host-reader", daemon=True).start()


class StdioFrontend:
    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        stdin: Any = None,
        stdout: Any = None,
        engine_address: str | None = None,
        autostart: bool = True,
    ) -> None:
        self.name = name
        self.command = resolve_command(command)
        self.stdin = stdin if stdin is not None else sys.stdin.buffer
        self.stdout = stdout if stdout is not None else sys.stdout.buffer
        self.client = EngineClient(engine_address)
        self.autostart = autostart
        self.host_id = host_identity()
        self.proc: asyncio.subprocess.Process | None = None
        self.job: Any = None
        self._server_lock = asyncio.Lock()
        self._ordered: dict[str, asyncio.Queue[asyncio.Task[dict[str, Any]] | None]] = {}
        self._reconnecting: asyncio.Task[None] | None = None
        self._writer = _HostWriter(self.stdout)

    # ------------------------------------------------------------------ engine

    async def _connect(self) -> None:
        await self.client.connect(autostart=self.autostart)
        await self.client.hello(self.name, transport="stdio", host_id=self.host_id, command=shlex.join(self.command))

    def _ensure_reconnect(self) -> None:
        if self._reconnecting is None or self._reconnecting.done():
            self._reconnecting = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        while not self.client.connected:
            try:
                self.client = EngineClient(self.client.address)
                await self._connect()
                _err("reconnected to the engine")
                return
            except (EngineUnavailable, OSError) as exc:
                _err(f"engine still unavailable: {exc}")
                await asyncio.sleep(2.0)

    async def _decide(self, direction: str, message: Any) -> dict[str, Any]:
        if self.client.connected:
            try:
                return await self.client.inspect(direction, message)
            except EngineUnavailable:
                pass
        ref = new_ref()
        _err(f"engine unavailable — {'host' if direction == 'h2s' else 'server'} message blocked (ref {ref})")
        self._ensure_reconnect()
        return fail_closed(direction, message, ref).to_json()

    # ------------------------------------------------------------------ output

    async def _to_server(self, message: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        async with self._server_lock:
            with contextlib.suppress(ConnectionError, BrokenPipeError, RuntimeError):
                self.proc.stdin.write(dumps(message).encode("utf-8") + b"\n")
                await self.proc.stdin.drain()

    async def _apply(self, decision: dict[str, Any]) -> None:
        for msg in decision.get("to_server", []):
            await self._to_server(msg)
        for msg in decision.get("to_host", []):
            self._writer.send(msg)

    async def _releaser(self, direction: str) -> None:
        q = self._ordered[direction]
        while True:
            task = await q.get()
            if task is None:
                return
            try:
                decision = await task
            except Exception as exc:  # pragma: no cover - _decide does not raise
                _err(f"internal error: {exc}")
                continue
            await self._apply(decision)

    def _submit(self, direction: str, message: Any) -> None:
        self._ordered[direction].put_nowait(asyncio.create_task(self._decide(direction, message)))

    # ------------------------------------------------------------------ pumps

    async def _pump_host(self, lines: asyncio.Queue[bytes | None]) -> None:
        while True:
            line = await lines.get()
            if line is None:
                return
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                self._writer.send(error_response(None, make_error(PARSE_ERROR, "Parse error")))
                continue
            self._submit("h2s", message)

    async def _pump_server(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            try:
                line = await self.proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                _err("server wrote a line longer than the limit; dropped")
                continue
            if not line:
                return
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                _err("server wrote a non-JSON line to stdout; dropped")
                continue
            self._submit("s2h", message)

    # ------------------------------------------------------------------ lifecycle

    async def _spawn_server(self) -> None:
        kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": None,
            "limit": STREAM_LIMIT,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        self.proc = await asyncio.create_subprocess_exec(*self.command, **kwargs)
        if sys.platform == "win32":
            try:
                from .winjob import KillOnCloseJob

                self.job = KillOnCloseJob()
                self.job.assign(self.proc.pid)
            except OSError as exc:
                _err(f"could not place the server in a Job Object: {exc}")

    async def _stop_server(self, grace: float = 5.0) -> int:
        proc = self.proc
        if proc is None:
            return 0
        if proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
        try:
            return await asyncio.wait_for(proc.wait(), grace)
        except asyncio.TimeoutError:
            pass
        self._signal_tree(signal.SIGTERM)
        try:
            return await asyncio.wait_for(proc.wait(), grace)
        except asyncio.TimeoutError:
            self._signal_tree(getattr(signal, "SIGKILL", signal.SIGTERM))
            return await proc.wait()

    def _signal_tree(self, sig: int) -> None:
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(proc.pid, sig)

    async def run(self) -> int:
        try:
            await self._connect()
        except (EngineUnavailable, OSError) as exc:
            _err(f"cannot reach or start the engine ({exc}); all traffic will be blocked until it is available")
            self._ensure_reconnect()
        try:
            await self._spawn_server()
        except OSError as exc:
            _err(f"cannot start the server {self.command[0]!r}: {exc}")
            return 127
        self._writer.start()
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if sig is not None:
                with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.add_signal_handler(sig, stop.set)

        self._ordered = {"h2s": asyncio.Queue(), "s2h": asyncio.Queue()}
        releasers = [asyncio.create_task(self._releaser(d)) for d in ("h2s", "s2h")]
        lines: asyncio.Queue[bytes | None] = asyncio.Queue()
        _start_host_reader(loop, self.stdin, lines)
        host = asyncio.create_task(self._pump_host(lines))
        server = asyncio.create_task(self._pump_server())
        stopper = asyncio.create_task(stop.wait())
        await asyncio.wait({host, server, stopper}, return_when=asyncio.FIRST_COMPLETED)

        if host.done() and not server.done():
            # host closed stdin: let in-flight host messages reach the server, then close it
            self._ordered["h2s"].put_nowait(None)
            await releasers[0]
            code = await self._stop_server()
            if not server.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(server), 5)
        elif server.done():
            code = await self._stop_server(grace=1.0)
        else:
            code = await self._stop_server(grace=2.0)
        for direction in ("h2s", "s2h"):
            self._ordered[direction].put_nowait(None)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*releasers, return_exceptions=True), 10)
        for task in (host, server, stopper, self._reconnecting):
            if task is not None and not task.done():
                task.cancel()
        self._writer.q.put(None)
        self._writer.join(timeout=5)
        await self.client.close()
        if self.job is not None:
            self.job.close()
        return code if isinstance(code, int) and code >= 0 else 0


def run_stdio(name: str, command: list[str], new_loop: Callable[[], asyncio.AbstractEventLoop]) -> int:
    loop = new_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(StdioFrontend(name, command).run())
    finally:
        loop.close()
