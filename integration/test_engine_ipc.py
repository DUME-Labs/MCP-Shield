"""The engine in-process: IPC handshake, front-end protocol, commands, fail-closed."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from mcp_shield.config import SecurityConfig
from mcp_shield.engine import ipc
from mcp_shield.engine.client import EngineClient, EngineUnavailable
from mcp_shield.engine.core import fail_closed
from mcp_shield.engine.server import Engine
from mcp_shield.errors import BLOCK_CODE
from mcp_shield.store import Store


async def start_engine(classifier, shield_home):
    engine = Engine(
        SecurityConfig(),
        classifier=classifier,
        store=Store(shield_home / "data" / "e.db"),
        dashboard=False,
        foreground=True,
    )
    await engine.start()
    return engine


def test_frames_round_trip(arun):
    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(ipc.encode_frame({"t": "x", "v": "ü" * 3}))
        reader.feed_eof()
        return await ipc.read_frame(reader)

    assert arun(go()) == {"t": "x", "v": "üüü"}


def test_engine_session(arun, classifier, shield_home):
    async def go():
        engine = await start_engine(classifier, shield_home)
        client = EngineClient()
        await client.connect(autostart=False)
        hello = await client.hello("filesystem", host_id="host:1", command="npx server")
        assert hello["t"] == "hello_ok"
        lst = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        d = await client.inspect("h2s", lst)
        assert d["to_server"] == [lst]
        d = await client.inspect(
            "s2h",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {"name": "read_file", "description": "Read a file."},
                        {"name": "write_file", "description": "Write a file."},
                    ]
                },
            },
        )
        assert [t["name"] for t in d["to_host"][0]["result"]["tools"]] == ["read_file"]
        status = await client.command("status")
        assert status["frontends"][0]["server"] == "filesystem" and status["classifier"]["available"]
        pins = await client.command("pins", server="filesystem")
        assert {p["name"] for p in pins["items"]} == {"read_file", "write_file"}
        assert (await client.command("accept_pins", server="filesystem"))["accepted"] == 2
        with pytest.raises(RuntimeError):
            await client.command("no_such_command")
        # concurrent inspections come back matched to their sequence numbers
        msgs = [
            {
                "jsonrpc": "2.0",
                "id": 100 + i,
                "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": f"/f{i}"}},
            }
            for i in range(10)
        ]
        results = await asyncio.gather(*(client.inspect("h2s", m) for m in msgs))
        assert [r["to_server"][0]["id"] for r in results] == [m["id"] for m in msgs]
        await client.close()
        await engine.stop()

    arun(go())


def test_wrong_secret_refused(arun, classifier, shield_home):
    async def go():
        engine = await start_engine(classifier, shield_home)
        reader, writer = await ipc.open_connection()
        with pytest.raises((ipc.AuthError, asyncio.IncompleteReadError, ConnectionError)):
            await ipc.client_handshake(reader, writer, b"wrong" * 8)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        await asyncio.sleep(0.05)
        refused = [e for e in engine.inspector.store.events() if e["kind"] == "ipc"]
        await engine.stop()
        return refused

    assert arun(go()), "a refused IPC client is logged"


def test_engine_down_means_unavailable(arun):
    async def go():
        client = EngineClient()
        with pytest.raises(EngineUnavailable):
            await client.connect(autostart=False)

    arun(go())


def test_single_instance(arun, classifier, shield_home):
    async def go():
        engine = await start_engine(classifier, shield_home)
        second = Engine(
            SecurityConfig(), classifier=classifier, store=Store(shield_home / "data" / "f.db"), dashboard=False
        )
        with pytest.raises(ipc.IPCError):
            await second.start()
        await engine.stop()

    arun(go())


class TestFailClosed:
    def test_host_request_answered_with_stub(self):
        d = fail_closed("h2s", {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {}}, "0badc0de")
        assert d.to_host[0]["error"]["code"] == BLOCK_CODE and not d.to_server

    def test_server_response_becomes_stub(self):
        d = fail_closed("s2h", {"jsonrpc": "2.0", "id": 4, "result": {"content": []}}, "0badc0de")
        assert d.to_host[0]["id"] == 4 and "error" in d.to_host[0]

    def test_server_requests_declined_notifications_dropped(self):
        d = fail_closed("s2h", {"jsonrpc": "2.0", "id": 9, "method": "elicitation/create", "params": {}}, "r")
        assert d.to_server[0]["result"] == {"action": "decline"}
        assert fail_closed("s2h", {"jsonrpc": "2.0", "method": "notifications/progress"}, "r").to_host == []
        assert fail_closed("h2s", {"jsonrpc": "2.0", "method": "notifications/cancelled"}, "r").to_server == []


@pytest.mark.posix
def test_long_home_gets_a_short_private_socket(tmp_path, monkeypatch):
    """AF_UNIX paths are limited to 104 bytes on macOS; a long home must still yield a usable socket."""
    import os
    import stat

    long_home = tmp_path / ("x" * 60) / ("y" * 60)
    monkeypatch.setenv("MCP_SHIELD_HOME", str(long_home))
    path = ipc.endpoint()
    assert len(path.encode()) <= ipc.MAX_SOCKET_PATH
    folder = os.path.dirname(path)
    info = os.lstat(folder)
    assert info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700
    monkeypatch.setenv("MCP_SHIELD_HOME", str(long_home) + "-other")
    assert ipc.endpoint() != path, "different homes must not share a socket"
