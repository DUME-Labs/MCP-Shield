"""Talking to the engine from a front-end or the CLI, starting it when needed (02 §1)."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..paths import data_dir, ensure_private_dir
from . import ipc


class EngineUnavailable(ConnectionError):
    """The engine could not be reached or started — the caller fails closed."""


def _python_for_background() -> str:
    exe = Path(sys.executable)
    if sys.platform == "win32":
        windowless = exe.with_name("pythonw.exe")
        if windowless.exists():
            return str(windowless)
    return str(exe)


def spawn_engine() -> None:
    """Start the engine detached from the calling front-end and its host."""
    ensure_private_dir(data_dir())
    args = [_python_for_background(), "-m", "mcp_shield", "engine", "--background"]
    log = open(data_dir() / "engine.log", "ab")  # noqa: SIM115 - handed to the child
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": log,
        "close_fds": True,
        "cwd": str(data_dir()),
    }
    try:
        if sys.platform == "win32":
            base = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            try:  # leave the host's Job Object so closing the host does not kill the engine
                subprocess.Popen(args, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
            except OSError:
                subprocess.Popen(args, creationflags=base, **kwargs)
        else:
            subprocess.Popen(args, start_new_session=True, **kwargs)
    finally:
        log.close()


class EngineClient:
    def __init__(self, address: str | None = None) -> None:
        self.address = address or ipc.endpoint()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._futures: dict[Any, asyncio.Future[Any]] = {}
        self._seq = itertools.count(1)
        self._lock = asyncio.Lock()
        self.connected = False
        self.hello_info: dict[str, Any] = {}

    async def connect(self, *, autostart: bool = True, timeout: float = 45.0) -> None:
        try:
            await self._open()
            return
        except (OSError, ipc.IPCError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            first_error: Exception = exc
        if not autostart:
            raise EngineUnavailable(f"engine not running ({first_error})")
        spawn_engine()
        deadline = time.monotonic() + timeout
        last: Exception = first_error
        while time.monotonic() < deadline:
            await asyncio.sleep(0.25)
            try:
                await self._open()
                return
            except (OSError, ipc.IPCError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
                last = exc
        raise EngineUnavailable(
            f"engine did not start within {timeout:.0f} s ({last}); see {data_dir() / 'engine.log'}"
        )

    async def _open(self) -> None:
        reader, writer = await ipc.open_connection(self.address)
        try:
            await ipc.client_handshake(reader, writer, ipc.load_secret(create=False))
        except BaseException:
            writer.close()
            raise
        self._reader, self._writer = reader, writer
        self.connected = True
        self._read_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                frame = await ipc.read_frame(self._reader)
                kind = frame.get("t")
                key: Any = None
                if kind == "verdict":
                    key = ("v", frame.get("seq"))
                elif kind == "cmd_result":
                    key = ("c", frame.get("id"))
                elif kind == "hello_ok":
                    key = ("h", 0)
                fut = self._futures.pop(key, None)
                if fut is not None and not fut.done():
                    fut.set_result(frame)
        except (asyncio.IncompleteReadError, ConnectionError, ipc.IPCError, ValueError, OSError):
            pass
        finally:
            self.connected = False
            for fut in self._futures.values():
                if not fut.done():
                    fut.set_exception(EngineUnavailable("engine connection lost"))
            self._futures.clear()

    async def _request(self, key: Any, frame: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        if not self.connected or self._writer is None:
            raise EngineUnavailable("not connected to the engine")
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._futures[key] = fut
        try:
            async with self._lock:
                await ipc.write_frame(self._writer, frame)
        except (ConnectionError, OSError) as exc:
            self._futures.pop(key, None)
            raise EngineUnavailable(str(exc)) from exc
        try:
            return dict(await asyncio.wait_for(fut, timeout))
        except asyncio.TimeoutError as exc:
            self._futures.pop(key, None)
            raise EngineUnavailable("engine did not answer in time") from exc

    async def hello(
        self,
        server: str,
        *,
        transport: str = "stdio",
        host_id: str = "",
        command: str | None = None,
        remote: bool = False,
    ) -> dict[str, Any]:
        self.hello_info = {
            "t": "hello",
            "server": server,
            "transport": transport,
            "host_id": host_id,
            "command": command,
            "remote": remote,
            "pid": os.getpid(),
        }
        return await self._request(("h", 0), self.hello_info, 30)

    async def inspect(self, direction: str, message: Any, timeout: float = 300.0) -> dict[str, Any]:
        seq = next(self._seq)
        frame = await self._request(("v", seq), {"t": "msg", "seq": seq, "dir": direction, "message": message}, timeout)
        return dict(frame["decision"])

    async def command(self, cmd: str, timeout: float = 60.0, **args: Any) -> Any:
        cid = next(self._seq)
        frame = await self._request(("c", cid), {"t": "cmd", "id": cid, "cmd": cmd, "args": args}, timeout)
        if not frame.get("ok"):
            raise RuntimeError(frame.get("error", "command failed"))
        return frame.get("result")

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(BaseException):
                await self._read_task
        self.connected = False


async def call_engine(cmd: str, *, autostart: bool = False, **args: Any) -> Any:
    """One-shot command for the CLI."""
    client = EngineClient()
    await client.connect(autostart=autostart)
    try:
        return await client.command(cmd, **args)
    finally:
        await client.close()
