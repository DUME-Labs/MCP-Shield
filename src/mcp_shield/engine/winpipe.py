# mypy: disable-error-code="attr-defined"
"""Windows named-pipe server with an owner-only DACL (02 §1).

asyncio's own ``start_serving_pipe`` creates pipe instances with the default security
descriptor, which grants read access to Everyone. This module creates every instance
through ``CreateNamedPipeW`` with a DACL that admits only the current user's SID and
with ``PIPE_REJECT_REMOTE_CLIENTS``, then drives it with the Proactor loop exactly as
asyncio does. The HMAC challenge–response in :mod:`.ipc` is the second, independent
line: a process that wins the race for the pipe name still cannot authenticate.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import sys
from collections.abc import Awaitable, Callable
from ctypes import wintypes
from typing import Any

# Windows-only: imported only on Windows, and this form makes type checkers skip the module elsewhere.
assert sys.platform == "win32", "winpipe is Windows-only"

import _winapi  # type: ignore[import-not-found]  # noqa: E402 - after the platform guard on purpose
from asyncio import windows_events, windows_utils  # noqa: E402

log = logging.getLogger(__name__)

PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
TOKEN_QUERY = 0x0008
TOKEN_USER_CLASS = 1
SDDL_REVISION_1 = 1
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.LocalFree.argtypes = [ctypes.c_void_p]
_kernel32.LocalFree.restype = ctypes.c_void_p
_kernel32.CreateNamedPipeW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(SECURITY_ATTRIBUTES),
]
_kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
_advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
_advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
_advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(wintypes.ULONG),
]


def _check(ok: Any, what: str) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), what)


def current_user_sid() -> str:
    token = wintypes.HANDLE()
    _check(
        _advapi32.OpenProcessToken(_kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)), "OpenProcessToken"
    )
    try:
        size = wintypes.DWORD(0)
        _advapi32.GetTokenInformation(token, TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        _check(
            _advapi32.GetTokenInformation(token, TOKEN_USER_CLASS, buf, size, ctypes.byref(size)), "GetTokenInformation"
        )
        sid = SID_AND_ATTRIBUTES.from_buffer(buf).Sid
        text = wintypes.LPWSTR()
        _check(_advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)), "ConvertSidToStringSidW")
        try:
            return str(text.value)
        finally:
            _kernel32.LocalFree(text)
    finally:
        _kernel32.CloseHandle(token)


class OwnerOnlySecurity:
    """A self-relative security descriptor: protected DACL, full access for one SID only."""

    def __init__(self, sid: str | None = None) -> None:
        self.sid = sid or current_user_sid()
        self._sd = ctypes.c_void_p()
        sddl = f"D:P(A;;GA;;;{self.sid})"
        _check(
            _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, SDDL_REVISION_1, ctypes.byref(self._sd), None
            ),
            "ConvertStringSecurityDescriptorToSecurityDescriptorW",
        )
        self.attributes = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), self._sd, False)

    def free(self) -> None:
        if self._sd:
            _kernel32.LocalFree(self._sd)
            self._sd = ctypes.c_void_p()


class OwnerOnlyPipeServer(windows_events.PipeServer):  # type: ignore[misc]
    def __init__(self, address: str) -> None:
        self._security = OwnerOnlySecurity()
        super().__init__(address)

    def _server_pipe_handle(self, first: bool) -> Any:
        if self.closed():
            return None
        flags = _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED
        if first:
            flags |= _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        handle = _kernel32.CreateNamedPipeW(
            self._address,
            flags,
            _winapi.PIPE_TYPE_MESSAGE | _winapi.PIPE_READMODE_MESSAGE | _winapi.PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            _winapi.PIPE_UNLIMITED_INSTANCES,
            windows_utils.BUFSIZE,
            windows_utils.BUFSIZE,
            _winapi.NMPWAIT_WAIT_FOREVER,
            ctypes.byref(self._security.attributes),
        )
        if handle is None or handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error(), f"CreateNamedPipeW({self._address})")
        pipe = windows_utils.PipeHandle(handle)
        self._free_instances.add(pipe)
        return pipe

    def close(self) -> None:
        super().close()
        if getattr(self, "_security", None) is not None:
            self._security.free()

    __del__ = close


async def serve_pipe(
    handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]], address: str, limit: int
) -> Callable[[], None]:
    """Serve ``handler`` on ``address``; returns a close function. Mirrors asyncio's accept loop."""
    loop = asyncio.get_running_loop()
    if not isinstance(loop, windows_events.ProactorEventLoop):
        raise RuntimeError("named pipes need the Proactor event loop")
    try:
        server = OwnerOnlyPipeServer(address)
    except PermissionError as exc:
        from .ipc import IPCError

        raise IPCError(f"an engine (or another process) already owns {address}") from exc

    def factory() -> asyncio.StreamReaderProtocol:
        reader = asyncio.StreamReader(limit=limit, loop=loop)
        return asyncio.StreamReaderProtocol(reader, handler, loop=loop)

    def loop_accept_pipe(f: Any = None) -> None:
        pipe = None
        try:
            if f:
                pipe = f.result()
                server._free_instances.discard(pipe)
                if server.closed():
                    pipe.close()
                    return
                protocol = factory()
                loop._make_duplex_pipe_transport(pipe, protocol, extra={"addr": address})  # type: ignore[attr-defined]
            pipe = server._get_unconnected_pipe()
            if pipe is None:
                return
            f = loop._proactor.accept_pipe(pipe)  # type: ignore[attr-defined]
        except BrokenPipeError:
            if pipe and pipe.fileno() != -1:
                pipe.close()
            loop.call_soon(loop_accept_pipe)
        except OSError as exc:
            if pipe and pipe.fileno() != -1:
                log.warning("pipe accept failed: %s", exc)
                pipe.close()
            loop.call_soon(loop_accept_pipe)
        except asyncio.CancelledError:
            if pipe:
                pipe.close()
        else:
            server._accept_pipe_future = f
            f.add_done_callback(loop_accept_pipe)

    loop.call_soon(loop_accept_pipe)
    return server.close


async def connect_pipe(address: str, limit: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=limit, loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport, _ = await loop.create_pipe_connection(lambda: protocol, address)  # type: ignore[attr-defined]
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer
