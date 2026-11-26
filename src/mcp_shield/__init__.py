"""MCP-Shield — a semantic LLM firewall for the Model Context Protocol.

Stage 1 ships the proxy (``mcp-shield proxy``: one engine per OS user, thin stdio
front-ends and HTTP listeners), the Tier 1–3 cascade, session taint tracking, capability
pinning with first-contact quarantine, Spotlighting, the host prompt rules, ``doctor``,
``install`` and the local dashboard. The ``@shield`` decorator is Stage 2.
"""

from ._version import __version__
from .config import ExplainerProvider, SecurityConfig
from .errors import (
    SecurityError,
    SecurityInjectionError,
    SecurityPolicyError,
    SecurityTaintViolationError,
    SecurityTamperingError,
)

__all__ = [
    "ExplainerProvider",
    "SecurityConfig",
    "SecurityError",
    "SecurityInjectionError",
    "SecurityPolicyError",
    "SecurityTaintViolationError",
    "SecurityTamperingError",
    "__version__",
]
