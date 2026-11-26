"""Block reference IDs (02 §8).

A random 32-bit value shown as 8 hex digits: never sequential (that would count
blocks), never structured (that would name the tier), never derived from content
(an attacker could hash candidate payloads). A lookup key, not a secret.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable


def new_ref(exists: Callable[[str], bool] | None = None) -> str:
    """Return a fresh reference; redraw on the rare collision with ``exists``."""
    for _ in range(16):
        ref = f"{secrets.randbits(32):08x}"
        if exists is None or not exists(ref):
            return ref
    raise RuntimeError("could not draw a unique reference ID")
