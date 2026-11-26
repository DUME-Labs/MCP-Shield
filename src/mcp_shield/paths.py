"""Where MCP-Shield keeps things (07 §1).

* cache   — only what can be downloaded again (classifier weights).
* data    — state that must survive: pins, the event store, the dashboard token,
            the engine secret, the audit file.
* runtime — the engine's Unix socket (POSIX); Windows uses a named pipe instead.
* config  — ``mcp-shield.toml``.

``MCP_SHIELD_HOME`` redirects all four under one folder (tests, portable installs).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir, user_runtime_dir

APP = "mcp-shield"
ENV_HOME = "MCP_SHIELD_HOME"


def _override() -> Path | None:
    value = os.environ.get(ENV_HOME)
    return Path(value).expanduser().resolve() if value else None


def cache_dir() -> Path:
    base = _override()
    return base / "cache" if base else Path(user_cache_dir(APP, appauthor=False))


def data_dir() -> Path:
    base = _override()
    return base / "data" if base else Path(user_data_dir(APP, appauthor=False))


def runtime_dir() -> Path:
    base = _override()
    if base:
        return base / "run"
    if sys.platform == "win32":
        return data_dir() / "run"
    return Path(user_runtime_dir(APP, appauthor=False))


def config_path() -> Path:
    base = _override()
    return (base / "config" if base else Path(user_config_dir(APP, appauthor=False))) / "mcp-shield.toml"


def models_dir() -> Path:
    return cache_dir() / "models"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` readable only by the current user (0700 on POSIX).

    On Windows the per-user profile folders (``%LOCALAPPDATA%``) already carry an ACL
    that admits only the user, SYSTEM and administrators, and new files inherit it.
    """
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def write_private_file(path: Path, data: bytes) -> None:
    """Write ``data`` so only the current user can read it (0600 on POSIX)."""
    ensure_private_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    if os.name == "posix":
        os.chmod(path, 0o600)
