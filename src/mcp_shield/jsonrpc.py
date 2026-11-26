"""JSON-RPC 2.0 helpers: parsing, message kinds and error responses."""

from __future__ import annotations

import json
from typing import Any

from .protocol import INVALID_REQUEST

REQUEST = "request"
NOTIFICATION = "notification"
RESPONSE = "response"
ERROR = "error"

Message = dict[str, Any]
MessageId = str | int


def loads(raw: str | bytes) -> Any:
    """Parse one JSON text. Raises ``ValueError`` on invalid JSON."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def dumps(message: Any) -> str:
    """Serialise compactly on one line (JSON escapes newlines inside strings)."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":"))


def kind_of(message: Any) -> str | None:
    """Classify a message; ``None`` when it is not a single valid JSON-RPC 2.0 object."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return None
    has_id = "id" in message
    if has_id and not _valid_id(message["id"]):
        return None
    if "method" in message:
        if not isinstance(message["method"], str) or not message["method"]:
            return None
        params = message.get("params")
        if params is not None and not isinstance(params, (dict, list)):
            return None
        return REQUEST if has_id else NOTIFICATION
    if has_id and "result" in message and "error" not in message:
        return RESPONSE
    if has_id and "error" in message and "result" not in message:
        err = message["error"]
        if isinstance(err, dict) and isinstance(err.get("code"), int) and isinstance(err.get("message"), str):
            return ERROR
    return None


def _valid_id(value: Any) -> bool:
    return value is None or (isinstance(value, (str, int)) and not isinstance(value, bool))


def shape_problem(message: Any) -> str | None:
    """Describe why ``message`` is not acceptable, or return ``None`` when it is."""
    if isinstance(message, list):
        return "JSON-RPC batch arrays are not allowed"
    if kind_of(message) is None:
        return "not a valid JSON-RPC 2.0 message"
    return None


def error_response(msg_id: MessageId | None, error: dict[str, Any]) -> Message:
    return {"jsonrpc": "2.0", "id": msg_id, "error": error}


def make_error(code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return err


def invalid_request(msg_id: MessageId | None, why: str) -> Message:
    return error_response(msg_id, make_error(INVALID_REQUEST, why))


def params_of(message: Message) -> dict[str, Any]:
    params = message.get("params")
    return params if isinstance(params, dict) else {}


def meta_of(message: Message) -> dict[str, Any]:
    meta = params_of(message).get("_meta")
    return meta if isinstance(meta, dict) else {}
