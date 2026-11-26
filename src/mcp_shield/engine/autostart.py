"""Per-user login item for the engine, registered only when HTTP entries are routed (02 §1).

macOS: a launchd agent · Linux: a ``systemd --user`` unit · Windows: a per-user logon
entry (``HKCU\\…\\Run``, no administrator rights needed). ``install`` registers it after
confirmation; ``uninstall`` removes it. Each function returns what it did, so the CLI can
print it, and ``dry_run`` returns the plan without touching the system.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

LABEL = "io.mcp-shield.engine"
LAUNCHCTL = "/bin/launchctl"
UNIT = "mcp-shield-engine.service"
RUN_VALUE = "MCP-Shield Engine"


def engine_command() -> list[str]:
    exe = Path(sys.executable)
    if sys.platform == "win32" and exe.with_name("pythonw.exe").exists():
        exe = exe.with_name("pythonw.exe")
    return [str(exe), "-m", "mcp_shield", "engine", "--foreground"]


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k == "MCP_SHIELD_HOME"}


def launchd_plist(command: list[str]) -> str:
    args = "\n".join(f"    <string>{escape(a)}</string>" for a in command)
    env = "".join(f"    <key>{escape(k)}</key><string>{escape(v)}</string>\n" for k, v in _env().items())
    env_block = f"  <key>EnvironmentVariables</key>\n  <dict>\n{env}  </dict>\n" if env else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        f"  <key>Label</key><string>{LABEL}</string>\n"
        f"  <key>ProgramArguments</key>\n  <array>\n{args}\n  </array>\n{env_block}"
        "  <key>RunAtLoad</key><true/>\n  <key>KeepAlive</key><false/>\n"
        "  <key>ProcessType</key><string>Background</string>\n"
        "</dict>\n</plist>\n"
    )


def systemd_unit(command: list[str]) -> str:
    env = "".join(f"Environment={k}={shlex.quote(v)}\n" for k, v in _env().items())
    return (
        "[Unit]\nDescription=MCP-Shield engine (per-user)\n\n"
        f"[Service]\nType=simple\nExecStart={shlex.join(command)}\n{env}Restart=on-failure\nRestartSec=5\n\n"
        "[Install]\nWantedBy=default.target\n"
    )


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / UNIT


def register(dry_run: bool = False) -> str:
    command = engine_command()
    if sys.platform == "darwin":
        path = _plist_path()
        if dry_run:
            return f"write {path} and load it with launchctl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(launchd_plist(command), encoding="utf-8")
        uid = os.getuid()
        subprocess.run([LAUNCHCTL, "bootout", f"gui/{uid}", str(path)], check=False, capture_output=True)
        subprocess.run([LAUNCHCTL, "bootstrap", f"gui/{uid}", str(path)], check=False, capture_output=True)
        return f"launchd agent {path}"
    if sys.platform == "win32":
        value = subprocess.list2cmdline(command)
        if dry_run:
            return f"add HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run '{RUN_VALUE}' = {value}"
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, value)
        return f"logon entry HKCU\\…\\Run '{RUN_VALUE}'"
    path = _unit_path()
    if dry_run:
        return f"write {path} and enable it with systemctl --user"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(systemd_unit(command), encoding="utf-8")
    systemctl = shutil.which("systemctl")
    if systemctl:
        subprocess.run([systemctl, "--user", "daemon-reload"], check=False, capture_output=True)
        subprocess.run([systemctl, "--user", "enable", "--now", UNIT], check=False, capture_output=True)
    return f"systemd --user unit {path}"


def unregister(dry_run: bool = False) -> str | None:
    if sys.platform == "darwin":
        path = _plist_path()
        if not path.exists():
            return None
        if not dry_run:
            subprocess.run([LAUNCHCTL, "bootout", f"gui/{os.getuid()}", str(path)], check=False, capture_output=True)
            path.unlink()
        return f"removed launchd agent {path}"
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
            ) as key:
                winreg.QueryValueEx(key, RUN_VALUE)
                if not dry_run:
                    winreg.DeleteValue(key, RUN_VALUE)
        except FileNotFoundError:
            return None
        return f"removed logon entry '{RUN_VALUE}'"
    path = _unit_path()
    if not path.exists():
        return None
    if not dry_run:
        systemctl = shutil.which("systemctl")
        if systemctl:
            subprocess.run([systemctl, "--user", "disable", "--now", UNIT], check=False, capture_output=True)
        path.unlink()
    return f"removed systemd --user unit {path}"


def registered() -> bool:
    if sys.platform == "darwin":
        return _plist_path().exists()
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                winreg.QueryValueEx(key, RUN_VALUE)
            return True
        except FileNotFoundError:
            return False
    return _unit_path().exists()
