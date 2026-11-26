"""The four error classes and the opaque error the host receives (design doc 02 §8).

The error classes carry full detail inside the engine, the event store and the
dashboard. What crosses the wire to the host is always the same stub — one code,
one message, a random reference — so the *kind* of block is never inferable.
"""

from __future__ import annotations

from typing import Any

BLOCK_CODE = -32000
BLOCK_MESSAGE = "Request blocked by MCP-Shield policy (ref {ref})"


class SecurityError(Exception):
    """Base class. ``tier``/``cls``/``probability`` stay local; never put them on the wire."""

    kind = "security"

    def __init__(self, reason: str, *, tier: str = "", cls: str | None = None, probability: float | None = None):
        super().__init__(reason)
        self.reason = reason
        self.tier = tier
        self.cls = cls
        self.probability = probability


class SecurityPolicyError(SecurityError):
    """Tier 1: header/body mismatch, disallowed tool, argument policy, schema violation."""

    kind = "policy"


class SecurityInjectionError(SecurityError):
    """Tier 2/3: known-bad signature, DoS guard, or a classifier verdict other than benign."""

    kind = "injection"


class SecurityTaintViolationError(SecurityError):
    """Taint engine: a call carrying tainted content, or a call held under strict policy."""

    kind = "taint"


class SecurityTamperingError(SecurityError):
    """Pin store: a tool manifest changed after pinning."""

    kind = "tamper"


ERROR_CLASS_NAMES = {
    "policy": "SecurityPolicyError",
    "injection": "SecurityInjectionError",
    "dos": "SecurityInjectionError",
    "taint": "SecurityTaintViolationError",
    "hold": "SecurityTaintViolationError",
    "tamper": "SecurityTamperingError",
    "quarantine": "SecurityInjectionError",
}


def stub_error(ref: str) -> dict[str, Any]:
    """The only error body the host ever receives for a block."""
    return {"code": BLOCK_CODE, "message": BLOCK_MESSAGE.format(ref=ref)}


def is_stub_message(message: object) -> bool:
    """True when text imitates the stub — used to label server-returned look-alikes."""
    return isinstance(message, str) and message.startswith("Request blocked by MCP-Shield policy (ref ")
