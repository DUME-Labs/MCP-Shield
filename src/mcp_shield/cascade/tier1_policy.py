"""Tier 1 — Structural & Policy Gate (02 §2). Deterministic, no model, < 1 ms.

1. header allowlist (HTTP, before the body is read) — see :func:`header_precheck`
2. header ↔ body consistency (``Mcp-Method``, ``Mcp-Name``, ``MCP-Protocol-Version``,
   ``Mcp-Param-*``) — a mismatch is the spec's ``HeaderMismatch`` (-32020, HTTP 400)
3. JSON-RPC shape (no batches) and schema conformance against the pinned ``inputSchema``
4. argument policy (:mod:`.argpolicy`)
5. tool allowlist and privilege class (:mod:`.privilege`)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import protocol as P
from ..config import ToolPolicy
from ..jsonrpc import meta_of, params_of, shape_problem
from .argpolicy import ArgFinding, scan_arguments
from .schema import SchemaUnusable, validate_arguments


@dataclass
class Tier1Result:
    block: str | None = None
    findings: list[ArgFinding] = field(default_factory=list)
    schema_warning: str | None = None

    @property
    def annotated(self) -> bool:
        return any(not f.block for f in self.findings)


def check_shape(message: Any) -> str | None:
    return shape_problem(message)


# --------------------------------------------------------------------------- headers (HTTP mode)


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def header_precheck(headers: Mapping[str, str], name_allowed: Callable[[str, str], bool]) -> str | None:
    """Header allowlist, run before the body is read. Returns a block reason or ``None``."""
    h = _lower(headers)
    method = h.get(P.HDR_METHOD)
    if method is None:
        return None  # older client: Tier 1 applies the allowlist to the body instead
    if method == "tools/call" and P.HDR_NAME in h:
        try:
            name = P.decode_header_value(h[P.HDR_NAME])
        except ValueError:
            return None  # reported as a HeaderMismatch once the body is read
        if not name_allowed(method, name):
            return f"tool '{name}' is not on the allowlist"
    return None


def _uses_current_revision(h: dict[str, str], message: dict[str, Any]) -> bool:
    return h.get(P.HDR_PROTOCOL_VERSION) == P.CURRENT_REVISION or (
        meta_of(message).get(P.META_PROTOCOL_VERSION) == P.CURRENT_REVISION
    )


def header_body_mismatches(
    headers: Mapping[str, str], message: dict[str, Any], input_schema: dict[str, Any] | None = None
) -> list[str]:
    """Every way the routing headers disagree with the body (empty when consistent)."""
    h = _lower(headers)
    problems: list[str] = []
    method = message.get("method")
    current = _uses_current_revision(h, message)
    params = params_of(message)

    if P.HDR_METHOD in h:
        if h[P.HDR_METHOD] != method:
            problems.append(f"Mcp-Method '{h[P.HDR_METHOD]}' does not match body method '{method}'")
    elif current and isinstance(method, str):
        problems.append("Mcp-Method header missing")

    if isinstance(method, str) and method in P.NAME_FIELD:
        body_name = params.get(P.NAME_FIELD[method])
        if P.HDR_NAME in h:
            try:
                header_name = P.decode_header_value(h[P.HDR_NAME])
            except ValueError:
                problems.append("Mcp-Name header is not valid =?base64?…?= text")
            else:
                if header_name != body_name:
                    problems.append("Mcp-Name does not match the body")
        elif current:
            problems.append("Mcp-Name header missing")

    body_version = meta_of(message).get(P.META_PROTOCOL_VERSION)
    if P.HDR_PROTOCOL_VERSION in h and body_version is not None and h[P.HDR_PROTOCOL_VERSION] != body_version:
        problems.append("MCP-Protocol-Version does not match _meta protocolVersion")
    if P.HDR_PROTOCOL_VERSION not in h and body_version == P.CURRENT_REVISION:
        problems.append("MCP-Protocol-Version header missing")

    problems.extend(_param_header_problems(h, params, input_schema, method))
    return problems


def _param_header_problems(
    h: dict[str, str], params: dict[str, Any], input_schema: dict[str, Any] | None, method: Any
) -> list[str]:
    sent = {k[len(P.HDR_PARAM_PREFIX) :]: v for k, v in h.items() if k.startswith(P.HDR_PARAM_PREFIX)}
    if method != "tools/call":
        return [f"Mcp-Param-{k} sent on {method}" for k in sent]
    declared: dict[str, str] = {}
    props = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if isinstance(props, dict):
        for prop, schema in props.items():
            if isinstance(schema, dict) and isinstance(schema.get("x-mcp-header"), str):
                declared[schema["x-mcp-header"].lower()] = prop
    if input_schema is None:
        return []  # nothing pinned to compare against yet
    problems: list[str] = []
    raw_args = params.get("arguments")
    args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
    for header, prop in declared.items():
        if prop not in args:
            if header in sent:
                problems.append(f"Mcp-Param-{header} sent without argument '{prop}'")
            continue
        if header not in sent:
            problems.append(f"Mcp-Param-{header} missing for argument '{prop}'")
            continue
        try:
            value = P.decode_header_value(sent[header])
        except ValueError:
            problems.append(f"Mcp-Param-{header} is not valid =?base64?…?= text")
            continue
        if value != _header_form(args[prop]):
            problems.append(f"Mcp-Param-{header} does not match argument '{prop}'")
    for header in sent:
        if header not in declared:
            problems.append(f"Mcp-Param-{header} is not declared by the tool")
    return problems


def _header_form(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


# --------------------------------------------------------------------------- tool calls


def check_tool_call(
    arguments: Any,
    tool_def: dict[str, Any] | None,
    policy: ToolPolicy,
    rules: str,
    *,
    local: bool,
) -> Tier1Result:
    """Schema conformance + argument policy for one ``tools/call`` (allowlist checked by the caller)."""
    result = Tier1Result()
    if arguments is not None and not isinstance(arguments, dict):
        result.block = "arguments must be an object"
        return result
    schema = tool_def.get("inputSchema") if isinstance(tool_def, dict) else None
    try:
        violations = validate_arguments(schema if isinstance(schema, dict) else None, arguments or {})
    except SchemaUnusable as exc:
        violations = []
        result.schema_warning = f"server inputSchema is invalid, schema check skipped: {exc}"
    if violations:
        result.block = "schema violation: " + "; ".join(violations)
        return result
    result.findings = scan_arguments(
        arguments or {}, schema if isinstance(schema, dict) else None, policy, rules, local=local
    )
    blocking = [f for f in result.findings if f.block]
    if blocking:
        result.block = "argument policy: " + "; ".join(
            f"{f.path} ({f.field_type}): {f.kind} — {f.detail}" for f in blocking
        )
    return result
