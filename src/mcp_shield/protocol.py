"""MCP protocol facts the proxy depends on (spec revision 2026-07-28, plus 2025-11-25 peers).

Checked against modelcontextprotocol.io/specification/2026-07-28: the routing headers,
the ``=?base64?…?=`` header encoding, ``resultType``, the MRTR map shape, the
``_meta`` keys and the error codes.
"""

from __future__ import annotations

import base64
import binascii
import re

# --- spec revisions --------------------------------------------------------------
CURRENT_REVISION = "2026-07-28"
LEGACY_REVISION = "2025-11-25"

# --- _meta keys -------------------------------------------------------------------
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
TRACE_KEYS = ("traceparent", "tracestate", "baggage")

# --- HTTP headers (lower-case; header names are case-insensitive) -----------------
HDR_METHOD = "mcp-method"
HDR_NAME = "mcp-name"
HDR_PROTOCOL_VERSION = "mcp-protocol-version"
HDR_PARAM_PREFIX = "mcp-param-"
HDR_SESSION_LEGACY = "mcp-session-id"

# Methods whose request carries Mcp-Name, and the params field it mirrors.
NAME_FIELD = {"tools/call": "name", "resources/read": "uri", "prompts/get": "name"}
# Only these methods may answer with resultType "input_required" (MRTR).
INPUT_REQUIRED_METHODS = frozenset(NAME_FIELD)

RESULT_COMPLETE = "complete"
RESULT_INPUT_REQUIRED = "input_required"

# --- error codes ------------------------------------------------------------------
HEADER_MISMATCH = -32020
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
PARSE_ERROR = -32700
SAMPLING_USER_REJECTED = -1  # 2025-11-25: the error an unproxied host sends when the user declines

# --- methods ----------------------------------------------------------------------
LIST_METHODS = frozenset({"tools/list", "prompts/list", "resources/list", "resources/templates/list"})
PINNED_LIST_METHODS = {"tools/list": "tool", "prompts/list": "prompt", "resources/templates/list": "template"}
LEGACY_SERVER_REQUESTS = frozenset({"sampling/createMessage", "elicitation/create", "roots/list", "ping"})
LIST_CHANGED_NOTIFICATIONS = frozenset(
    {"notifications/tools/list_changed", "notifications/prompts/list_changed", "notifications/resources/list_changed"}
)

_B64_HEADER = re.compile(r"^=\?base64\?([A-Za-z0-9+/=]*)\?=$")
_PLAIN_SAFE = re.compile(r"^[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?$")


def decode_header_value(value: str) -> str:
    """Decode the spec's ``=?base64?<b64 of UTF-8>?=`` form; plain values pass through."""
    stripped = value.strip()
    match = _B64_HEADER.match(stripped)
    if not match:
        if stripped.startswith("=?base64?") and stripped.endswith("?="):
            raise ValueError(f"malformed base64 header value: {value[:40]!r}")
        return stripped
    try:
        return base64.b64decode(match.group(1), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"malformed base64 header value: {value[:40]!r}") from exc


def encode_header_value(value: str) -> str:
    """Encode a value for an Mcp-* header: plain printable ASCII as-is, otherwise base64 form."""
    if _PLAIN_SAFE.match(value) and not (value.startswith("=?base64?") and value.endswith("?=")):
        return value
    return "=?base64?" + base64.b64encode(value.encode("utf-8")).decode("ascii") + "?="
