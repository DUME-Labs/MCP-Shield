"""The engine's local channel (02 §1, "The engine's local channel").

* macOS / Linux — a Unix domain socket in the per-user runtime directory (mode 0600,
  directory 0700).
* Windows — a named pipe whose DACL admits only the current user and which rejects
  remote clients (``PIPE_REJECT_REMOTE_CLIENTS``).

Not a TCP port, so no web page can address it. Before any message flows both ends prove
knowledge of a per-user secret kept in the data directory (HMAC challenge–response in
both directions), so a process that grabbed the pipe name or socket path first learns
nothing and is refused. Frames are a 4-byte big-endian length followed by UTF-8 JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import struct
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..paths import ENV_HOME, data_dir, ensure_private_dir, runtime_dir, write_private_file

MAX_FRAME = 64 * 1024 * 1024
PROTOCOL_VERSION = 1
SECRET_FILE = "engine.secret"  # noqa: S105 - a file name, not a secret
HANDSHAKE_TIMEOUT = 10.0

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


class IPCError(ConnectionError):
    pass


class AuthError(IPCError):
    pass


# --------------------------------------------------------------------------- framing


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    header = await reader.readexactly(4)
    (size,) = struct.unpack(">I", header)
    if size > MAX_FRAME:
        raise IPCError(f"frame of {size} bytes exceeds the limit")
    payload = await reader.readexactly(size)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise IPCError("frame is not a JSON object")
    return obj


def encode_frame(obj: dict[str, Any]) -> bytes:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME:
        raise IPCError("frame too large")
    return struct.pack(">I", len(payload)) + payload


async def write_frame(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    writer.write(encode_frame(obj))
    await writer.drain()


# --------------------------------------------------------------------------- secret + endpoint


def load_secret(create: bool = True) -> bytes:
    path = data_dir() / SECRET_FILE
    if path.exists():
        data = path.read_bytes()
        if len(data) >= 32:
            return data
    if not create:
        raise AuthError(f"engine secret missing: {path}")
    data = secrets.token_bytes(32)
    write_private_file(path, data)
    return data


def endpoint() -> str:
    """The socket path (POSIX) or pipe name (Windows) for this OS user."""
    if sys.platform == "win32":
        user = os.environ.get("USERNAME", "user")
        home = os.environ.get(ENV_HOME, "")
        tag = hashlib.sha256(f"{user}|{home}".encode()).hexdigest()[:16]
        return rf"\\.\pipe\mcp-shield-{tag}"
    run = ensure_private_dir(runtime_dir())
    path = str(run / "engine.sock")
    if len(path.encode()) <= MAX_SOCKET_PATH:
        return path
    # AF_UNIX paths are limited to 104 bytes on macOS (108 on Linux): use a short socket in a
    # private per-user folder under /tmp, keyed by the long path so separate homes stay separate.
    return str(_short_socket_dir() / f"{hashlib.sha256(path.encode()).hexdigest()[:16]}.sock")


MAX_SOCKET_PATH = 100


def _short_socket_dir() -> Path:
    import stat
    import tempfile

    if sys.platform == "win32":  # Windows uses a named pipe, never a socket
        raise IPCError("Unix sockets are not used on Windows")
    folder = Path(tempfile.gettempdir()) / f"mcp-shield-{os.getuid()}"
    with contextlib.suppress(FileExistsError):
        folder.mkdir(mode=0o700)
    info = os.lstat(folder)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise IPCError(f"{folder} is not a private directory owned by this user; refusing to place the socket there")
    return folder


def _mac(secret: bytes, role: str, server_nonce: str, client_nonce: str) -> str:
    msg = f"mcp-shield-ipc|v{PROTOCOL_VERSION}|{role}|{server_nonce}|{client_nonce}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


async def server_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, secret: bytes) -> None:
    server_nonce = secrets.token_hex(16)
    await write_frame(writer, {"t": "challenge", "v": PROTOCOL_VERSION, "nonce": server_nonce})
    reply = await asyncio.wait_for(read_frame(reader), HANDSHAKE_TIMEOUT)
    client_nonce = str(reply.get("nonce", ""))
    if reply.get("t") != "auth" or len(client_nonce) < 32:
        raise AuthError("malformed authentication")
    if not hmac.compare_digest(str(reply.get("mac", "")), _mac(secret, "client", server_nonce, client_nonce)):
        raise AuthError("client failed the challenge")
    await write_frame(writer, {"t": "auth_ok", "mac": _mac(secret, "server", server_nonce, client_nonce)})


async def client_handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, secret: bytes) -> None:
    challenge = await asyncio.wait_for(read_frame(reader), HANDSHAKE_TIMEOUT)
    if challenge.get("t") != "challenge" or challenge.get("v") != PROTOCOL_VERSION:
        raise AuthError("unexpected greeting from the engine endpoint")
    server_nonce = str(challenge.get("nonce", ""))
    client_nonce = secrets.token_hex(16)
    await write_frame(
        writer, {"t": "auth", "nonce": client_nonce, "mac": _mac(secret, "client", server_nonce, client_nonce)}
    )
    reply = await asyncio.wait_for(read_frame(reader), HANDSHAKE_TIMEOUT)
    if reply.get("t") != "auth_ok" or not hmac.compare_digest(
        str(reply.get("mac", "")), _mac(secret, "server", server_nonce, client_nonce)
    ):
        raise AuthError("the process on the engine endpoint does not know the engine secret — refusing it")


# --------------------------------------------------------------------------- server


class Listener:
    """A listening endpoint. ``aclose`` also closes connections it accepted but nobody handled yet."""

    def __init__(self, close: Callable[[], None], server: asyncio.AbstractServer | None = None) -> None:
        self._close = close
        self._server = server

    def close(self) -> None:
        self._close()

    async def aclose(self) -> None:
        self._close()
        server = self._server
        if server is None:
            return
        close_clients = getattr(server, "close_clients", None)  # Python 3.13+
        if close_clients is not None:
            close_clients()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), 2)


async def start_server(handler: Handler, address: str | None = None) -> Listener:
    address = address or endpoint()
    if sys.platform == "win32":
        from .winpipe import serve_pipe

        return Listener(await serve_pipe(handler, address, MAX_FRAME + 8))
    path = Path(address)
    if path.exists():
        try:
            _, w = await asyncio.open_unix_connection(str(path))
            w.close()
            with contextlib.suppress(Exception):
                await w.wait_closed()
            raise IPCError(f"an engine is already listening on {path}")
        except (ConnectionRefusedError, FileNotFoundError):
            path.unlink(missing_ok=True)
    old = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(handler, str(path), limit=MAX_FRAME + 8)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)

    def close() -> None:
        server.close()
        path.unlink(missing_ok=True)

    return Listener(close, server)


async def open_connection(address: str | None = None) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    address = address or endpoint()
    if sys.platform == "win32":
        from .winpipe import connect_pipe

        return await connect_pipe(address, MAX_FRAME + 8)
    return await asyncio.open_unix_connection(address, limit=MAX_FRAME + 8)
