"""Host identity for the session key (02 §4): the process that launched the front-end.

Launcher shims between the host and the front-end (the ``mcp-shield`` console-script
wrapper, a ``python`` re-exec, ``uv``/``pipx``, a shell) are skipped, so two front-ends
launched by one host instance resolve to the same identity.
"""

from __future__ import annotations

import os

_SHIMS = frozenset(
    {
        "mcp-shield",
        "mcp-shield.exe",
        "python",
        "python.exe",
        "pythonw.exe",
        "python3",
        "py.exe",
        "py",
        "uv",
        "uv.exe",
        "uvx",
        "uvx.exe",
        "pipx",
        "pipx.exe",
        "sh",
        "bash",
        "zsh",
        "dash",
        "fish",
        "cmd.exe",
        "conhost.exe",
        "env",
    }
)


def _is_shim(name: str) -> bool:
    n = name.lower()
    return n in _SHIMS or (n.startswith("python3.") and not n.endswith(".app"))


def host_identity(max_hops: int = 6) -> str:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a dependency; keep a fallback anyway
        return f"ppid:{os.getppid()}"
    try:
        proc = psutil.Process().parent()
        hops = 0
        while proc is not None and hops < max_hops and _is_shim(proc.name()):
            parent = proc.parent()
            if parent is None:
                break
            proc = parent
            hops += 1
        if proc is None:
            return f"ppid:{os.getppid()}"
        return f"{proc.name()}:{proc.pid}:{int(proc.create_time())}"
    except (psutil.Error, OSError):
        return f"ppid:{os.getppid()}"
