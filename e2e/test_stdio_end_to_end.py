"""Real processes: `mcp-shield proxy` front-end → autostarted engine → fake stdio server.

This is the path a host uses: it launches the front-end with the server's command line,
the front-end starts the engine on first use, and every message goes through the engine.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import PROJECT_SRC, TESTS

pytestmark = pytest.mark.slow
FAKE = TESTS / "fakes" / "fake_mcp_server.py"


def env_for(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["MCP_SHIELD_HOME"] = str(home)
    env["PYTHONPATH"] = str(PROJECT_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def cli(home: Path, *args: str, timeout: float = 60, input: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mcp_shield", *args],
        env=env_for(home),
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input,
        encoding="utf-8",
    )


class Frontend:
    def __init__(self, home: Path, name: str = "fake") -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_shield", "proxy", "--name", name, "--", sys.executable, str(FAKE)],
            env=env_for(home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.lines: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.stderr: list[str] = []
        threading.Thread(target=self._read_err, daemon=True).start()
        self.next_id = 0

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in iter(self.proc.stdout.readline, b""):
            self.lines.put(line)

    def _read_err(self) -> None:
        assert self.proc.stderr is not None
        for line in iter(self.proc.stderr.readline, b""):
            self.stderr.append(line.decode("utf-8", "replace"))

    def send(self, method: str, params: dict | None = None) -> int:
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()
        return self.next_id

    def recv(self, msg_id: int, timeout: float = 60) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise AssertionError("front-end exited: " + "".join(self.stderr)) from None
                continue
            msg = json.loads(line)
            if msg.get("id") == msg_id:
                return msg
        raise AssertionError(f"no response to {msg_id}; stderr: {''.join(self.stderr)}")

    def request(self, method: str, params: dict | None = None) -> dict:
        return self.recv(self.send(method, params))

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=30)


@pytest.fixture
def home(tmp_path, tiny_model_dir):
    h = tmp_path / "e2e-home"
    r = cli(h, "pull-model", "--from", str(tiny_model_dir))
    assert r.returncode == 0, r.stderr
    yield h
    cli(h, "engine", "--stop")
    time.sleep(0.5)


def tool_names(resp: dict) -> list[str]:
    return [t["name"] for t in resp["result"]["tools"]]


def test_full_path(home):
    fe = Frontend(home)
    try:
        init = fe.request(
            "initialize",
            {"protocolVersion": "2025-11-25", "clientInfo": {"name": "e2e", "version": "1"}, "capabilities": {}},
        )
        assert init["result"]["serverInfo"]["name"] == "fake"
        assert tool_names(fe.request("tools/list")) == ["read_file"], "first contact: quarantined"

        show = cli(home, "pin", "--server", "fake", "--show")
        assert show.returncode == 0 and "write_file" in show.stdout and "pending review" in show.stdout
        accept = cli(home, "pin", "--server", "fake", "--accept", "--yes")
        assert accept.returncode == 0, accept.stderr
        assert tool_names(fe.request("tools/list")) == ["read_file", "write_file", "run_shell"]

        ok = fe.request("tools/call", {"name": "read_file", "arguments": {"path": "notes.md"}})
        text = ok["result"]["content"][0]["text"]
        assert text.startswith('[MCP-Shield: untrusted data from server "fake"') and "release notes" in text

        bad = fe.request("tools/call", {"name": "read_file", "arguments": {"path": "poison.md"}})
        assert bad["error"]["code"] == -32000 and "ref " in bad["error"]["message"]
        ref = bad["error"]["message"].split("ref ")[1].rstrip(")")
        events = cli(home, "events", "--ref", ref)
        assert json.loads(events.stdout.strip().splitlines()[-1])["kind"] == "injection"

        fe.request("tools/call", {"name": "read_file", "arguments": {"path": "secret.env"}})
        leak = fe.request(
            "tools/call",
            {
                "name": "write_file",
                "arguments": {
                    "path": "public.md",
                    "content": "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                },
            },
        )
        assert leak["error"]["code"] == -32000, "tainted content written elsewhere is blocked"

        status = cli(home, "engine", "--status")
        data = json.loads(status.stdout)
        assert data["classifier"]["available"] and data["frontends"][0]["server"] == "fake"
    finally:
        code = fe.close()
    assert code == 0
    assert any("non-JSON" in line for line in fe.stderr), "junk on the server's stdout is dropped, not forwarded"


def test_engine_unavailable_fails_closed(tmp_path):
    home = tmp_path / "broken-home"
    home.mkdir()
    (home / "data").mkdir()
    # an unusable model makes the engine fail closed rather than refuse to start
    fe = Frontend(home, "fake2")
    try:
        lst = fe.request("tools/list")
        assert lst["result"]["tools"] == [], "descriptions cannot be scanned: everything withheld"
        call = fe.request("tools/call", {"name": "read_file", "arguments": {"path": "notes.md"}})
        assert call["error"]["code"] == -32000
    finally:
        fe.close()
        cli(home, "engine", "--stop")
