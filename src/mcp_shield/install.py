"""``mcp-shield install`` / ``uninstall`` — route a host's MCP servers through the proxy (02 §1, 07 §3).

* every stdio entry becomes ``mcp-shield proxy --name <entry> -- <original command>``;
* every non-OAuth HTTP entry (loopback, or carrying a static ``Authorization`` / API-key
  header) is pointed at a new engine listener, recorded in ``mcp-shield.toml``;
* entries that probably use OAuth are reported and left untouched (02 §1, "Which servers
  the proxy can front"), as are legacy HTTP+SSE entries;
* the original file is backed up, and ``uninstall`` restores exactly the entries this
  tool changed, keeping any other edits made since.

Nothing here runs a server: the plan is printed in full and applied only on confirmation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .paths import config_path, data_dir, write_private_file

STATE_FILE = "install-state.json"
MARK = "# added by mcp-shield install"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
_AUTH_HEADERS = ("authorization", "x-api-key", "api-key", "x-auth-token")


def known_host_paths() -> dict[str, Path]:
    home = Path.home()
    if sys.platform == "win32":
        claude = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "darwin":
        claude = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        claude = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "Claude" / "claude_desktop_config.json"
    return {"claude-desktop": claude, "cursor": home / ".cursor" / "mcp.json"}


def resolve_host(host: str) -> Path:
    known = known_host_paths()
    return known[host] if host in known else Path(host).expanduser()


def shield_command() -> list[str]:
    exe = shutil.which("mcp-shield")
    if exe:
        return [str(Path(exe).resolve())]
    return [sys.executable, "-m", "mcp_shield"]


def is_wrapped(entry: dict[str, Any]) -> bool:
    command = str(entry.get("command", ""))
    args = [str(a) for a in entry.get("args") or []]
    name = Path(command).name.lower()
    if name.startswith("mcp-shield") and args[:1] == ["proxy"]:
        return True
    return "mcp_shield" in args and "proxy" in args


def _entry_url(entry: dict[str, Any]) -> str | None:
    for key in ("url", "serverUrl"):
        if isinstance(entry.get(key), str):
            return str(entry[key])
    return None


@dataclass
class Change:
    name: str
    action: str  # wrap-stdio | front-http | skip-wrapped | skip-oauth | skip-legacy-sse | skip-unknown
    before: dict[str, Any]
    after: dict[str, Any] | None = None
    listener: dict[str, Any] | None = None
    note: str = ""

    def describe(self) -> str:
        if self.action == "wrap-stdio":
            assert self.after is not None
            cmd = " ".join([str(self.after["command"]), *[str(a) for a in self.after.get("args", [])]])
            return f"[wrap]   {self.name}: {cmd}"
        if self.action == "front-http":
            assert self.listener is not None and self.after is not None
            return (
                f"[listen] {self.name}: {self.listener['upstream']} -> engine listener {self.listener['listen']} "
                f"(host now uses {_entry_url(self.after)})"
            )
        return f"[skip]   {self.name}: {self.note}"


@dataclass
class Plan:
    path: Path
    original: dict[str, Any]
    changes: list[Change] = field(default_factory=list)

    @property
    def effective(self) -> list[Change]:
        return [c for c in self.changes if c.action in ("wrap-stdio", "front-http")]

    @property
    def needs_login_item(self) -> bool:
        return any(c.action == "front-http" for c in self.changes)


def load_host_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"mcpServers": {}}
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    data.setdefault("mcpServers", {})
    return data


def _used_ports(text: str) -> set[int]:
    return {int(p) for p in re.findall(r"listen\s*=\s*\"[^\"]*:(\d+)\"", text)}


def plan_install(path: Path, *, base_port: int = 8801) -> Plan:
    data = load_host_config(path)
    plan = Plan(path, data)
    toml_text = config_path().read_text(encoding="utf-8") if config_path().exists() else ""
    used = _used_ports(toml_text)
    listener_ports = set(used)
    port = base_port
    shield = shield_command()
    for name, entry in (data.get("mcpServers") or {}).items():
        if not isinstance(entry, dict):
            plan.changes.append(Change(name, "skip-unknown", {}, note="entry is not an object"))
            continue
        if is_wrapped(entry):
            plan.changes.append(Change(name, "skip-wrapped", entry, note="already routed through MCP-Shield"))
            continue
        if isinstance(entry.get("command"), str):
            after = dict(entry)
            after["command"] = shield[0]
            after["args"] = [
                *shield[1:],
                "proxy",
                "--name",
                name,
                "--",
                entry["command"],
                *[str(a) for a in entry.get("args") or []],
            ]
            plan.changes.append(Change(name, "wrap-stdio", entry, after))
            continue
        url = _entry_url(entry)
        if url is None:
            plan.changes.append(Change(name, "skip-unknown", entry, note="neither a command nor a URL"))
            continue
        if str(entry.get("type", "")).lower() == "sse" or urlsplit(url).path.rstrip("/").endswith("/sse"):
            plan.changes.append(
                Change(name, "skip-legacy-sse", entry, note="legacy HTTP+SSE transport is not proxied; left unchanged")
            )
            continue
        host = urlsplit(url).hostname or ""
        if host in LOOPBACK and urlsplit(url).port in listener_ports:
            plan.changes.append(Change(name, "skip-wrapped", entry, note="already points at an engine listener"))
            continue
        headers = {str(k).lower() for k in (entry.get("headers") or {})}
        if host not in LOOPBACK and not headers & set(_AUTH_HEADERS):
            plan.changes.append(
                Change(
                    name,
                    "skip-oauth",
                    entry,
                    note=(
                        "remote server without a static credential — probably OAuth-protected; a proxy cannot be "
                        "transparent to it (Stage 2 OAuth proxy pattern). Left unchanged and NOT protected."
                    ),
                )
            )
            continue
        while port in used:
            port += 1
        used.add(port)
        listen = f"127.0.0.1:{port}"
        after = dict(entry)
        key = "url" if "url" in entry else "serverUrl"
        after[key] = f"http://{listen}{urlsplit(url).path or '/mcp'}"
        plan.changes.append(Change(name, "front-http", entry, after, {"name": name, "listen": listen, "upstream": url}))
    return plan


def _load_state() -> dict[str, Any]:
    try:
        return dict(json.loads((data_dir() / STATE_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    write_private_file(data_dir() / STATE_FILE, json.dumps(state, indent=2).encode("utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".mcp-shield.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def apply_install(plan: Plan) -> Path | None:
    """Back up, rewrite the host config, record what changed, add listeners. Returns the backup path."""
    if not plan.effective:
        return None
    backup = None
    if plan.path.exists():
        backup = plan.path.with_name(f"{plan.path.name}.mcp-shield-backup-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(plan.path, backup)
    data = json.loads(json.dumps(plan.original))
    state = _load_state()
    record = state.setdefault(str(plan.path.resolve()), {"entries": {}, "listeners": []})
    for change in plan.effective:
        assert change.after is not None
        data["mcpServers"][change.name] = change.after
        record["entries"][change.name] = {"original": change.before, "installed": change.after}
        if change.listener:
            _add_listener(change.listener, plan.path)
            record["listeners"].append(change.listener["name"])
    _write_json(plan.path, data)
    _save_state(state)
    return backup


def _add_listener(listener: dict[str, Any], host_path: Path) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = (
        f"\n{MARK} ({host_path.name}:{listener['name']})\n[[listeners]]\n"
        f"name = {json.dumps(listener['name'])}\nlisten = {json.dumps(listener['listen'])}\n"
        f"upstream = {json.dumps(listener['upstream'])}\n"
    )
    path.write_text(text + block, encoding="utf-8")


def _remove_listener(name: str) -> bool:
    path = config_path()
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"\n{re.escape(MARK)} \([^)]*:{re.escape(name)}\)\n\[\[listeners\]\]\n(?:[^\[\n#][^\n]*\n)*")
    new, n = pattern.subn("", text)
    if n:
        path.write_text(new, encoding="utf-8")
    return bool(n)


def plan_uninstall(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Entries to restore: only those still exactly as ``install`` left them."""
    state = _load_state().get(str(path.resolve()), {})
    data = load_host_config(path)
    out = []
    for name, rec in state.get("entries", {}).items():
        current = data["mcpServers"].get(name)
        if current == rec["installed"]:
            out.append((name, rec["original"]))
    return out


def apply_uninstall(path: Path) -> list[str]:
    state = _load_state()
    key = str(path.resolve())
    record = state.get(key, {"entries": {}, "listeners": []})
    restore = plan_uninstall(path)
    data = load_host_config(path)
    for name, original in restore:
        data["mcpServers"][name] = original
    if restore:
        _write_json(path, data)
    for name in record.get("listeners", []):
        _remove_listener(name)
    state.pop(key, None)
    _save_state(state)
    return [n for n, _ in restore]


def any_listeners_installed() -> bool:
    return any(rec.get("listeners") for rec in _load_state().values())


def find_unproxied() -> list[tuple[str, str, str]]:
    """``(host, entry, why)`` for every known-host entry not routed through MCP-Shield."""
    out = []
    listener_ports = _used_ports(config_path().read_text(encoding="utf-8")) if config_path().exists() else set()
    for host, path in known_host_paths().items():
        if not path.exists():
            continue
        try:
            data = load_host_config(path)
        except (ValueError, OSError):
            continue
        for name, entry in (data.get("mcpServers") or {}).items():
            if isinstance(entry, dict) and not is_wrapped(entry):
                url = _entry_url(entry)
                parts = urlsplit(url) if url else None
                if parts and (parts.hostname or "") in LOOPBACK and parts.port in listener_ports:
                    continue  # pointed at an engine listener
                out.append((host, name, "stdio" if "command" in entry else f"http {url}"))
    return out


def write_cursor_rules(host_path: Path, text: str) -> Path:
    root = host_path.parent  # …/.cursor
    target = root / "rules" / "mcp-shield.mdc"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "---\ndescription: MCP-Shield host rules — how to treat MCP tool output\nalwaysApply: true\n---\n\n" + text,
        encoding="utf-8",
    )
    return target
