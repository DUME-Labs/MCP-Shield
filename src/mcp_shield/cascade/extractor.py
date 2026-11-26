"""Untrusted-Segment Extractor (02 §3).

A deterministic walker over a JSON-RPC message that yields every string originating
from an untrusted party — tool result text, resource contents, descriptions, prompt
messages, MRTR input requests, server-authored errors and notifications, and call
arguments above a minimum length. No model is involved.

Each :class:`Segment` carries the classifier input (normalised text plus a sanitised
context prefix) and the location of the string in the message, so later stages can
wrap it in Spotlighting markers or truncate it without walking the message again.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..textnorm import context_prefix, decoded_base64_runs, has_hidden_characters, normalize

Loc = tuple[str | int, ...]

KIND_RESULT = "result"
KIND_DESCRIPTION = "description"
KIND_ARGUMENT = "argument"
KIND_INPUT_REQUEST = "input_request"
KIND_NOTIFICATION = "notification"
KIND_ERROR = "error"
KIND_INSTRUCTIONS = "instructions"

MIN_DESCRIPTION_LEN = 8
_SCHEMA_TEXT_KEYS = frozenset({"description", "title"})
_SCHEMA_VALUE_KEYS = frozenset({"default", "examples", "enum", "const"})


@dataclass
class Segment:
    path: str
    raw: str
    loc: Loc
    kind: str
    server: str
    tool: str
    spotlight: bool = False
    derived: bool = False
    text: str = field(init=False)
    prefix: str = field(init=False)
    sha256: str = field(init=False)
    hidden_chars: bool = field(init=False)

    def __post_init__(self) -> None:
        self.text = normalize(self.raw)
        self.prefix = context_prefix(self.server, self.tool, self.path)
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        self.hidden_chars = has_hidden_characters(self.raw)


@dataclass
class Blob:
    path: str
    sha256: str
    size: int


@dataclass
class Extraction:
    segments: list[Segment] = field(default_factory=list)
    blobs: list[Blob] = field(default_factory=list)

    def add(self, seg: Segment, *, second_pass: bool = True) -> None:
        if not seg.raw or not seg.raw.strip():
            return
        self.segments.append(seg)
        if second_pass:
            for i, decoded in enumerate(decoded_base64_runs(seg.text)):
                self.segments.append(
                    Segment(f"{seg.path}#base64[{i}]", decoded, seg.loc, seg.kind, seg.server, seg.tool, derived=True)
                )

    def add_blob(self, path: str, data: Any) -> None:
        raw = data.encode("utf-8") if isinstance(data, str) else b""
        self.blobs.append(Blob(path, hashlib.sha256(raw).hexdigest(), len(raw)))

    @property
    def total_chars(self) -> int:
        return sum(len(s.raw) for s in self.segments if not s.derived)


def _fmt(loc: Loc) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def string_leaves(obj: Any, loc: Loc = ()) -> Iterator[tuple[Loc, str]]:
    if isinstance(obj, str):
        yield loc, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from string_leaves(value, (*loc, str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from string_leaves(value, (*loc, i))


# --------------------------------------------------------------------------- content blocks


def _content_block(ex: Extraction, block: Any, loc: Loc, server: str, tool: str, *, spotlight: bool) -> None:
    if not isinstance(block, dict):
        return
    kind = block.get("type")
    if kind == "text" and isinstance(block.get("text"), str):
        ex.add(Segment(_fmt((*loc, "text")), block["text"], (*loc, "text"), KIND_RESULT, server, tool, spotlight))
    elif kind == "resource" and isinstance(block.get("resource"), dict):
        _resource_contents(ex, block["resource"], (*loc, "resource"), server, tool, spotlight=spotlight)
    elif kind in ("image", "audio"):
        ex.add_blob(_fmt((*loc, "data")), block.get("data"))
    elif kind == "resource_link":
        for key in ("description", "title", "name"):
            if isinstance(block.get(key), str):
                ex.add(Segment(_fmt((*loc, key)), block[key], (*loc, key), KIND_RESULT, server, tool))


def _resource_contents(ex: Extraction, item: Any, loc: Loc, server: str, tool: str, *, spotlight: bool) -> None:
    if not isinstance(item, dict):
        return
    if isinstance(item.get("text"), str):
        ex.add(Segment(_fmt((*loc, "text")), item["text"], (*loc, "text"), KIND_RESULT, server, tool, spotlight))
    if "blob" in item:
        ex.add_blob(_fmt((*loc, "blob")), item.get("blob"))


def _structured(ex: Extraction, value: Any, loc: Loc, server: str, tool: str) -> None:
    for leaf_loc, text in string_leaves(value, loc):
        ex.add(Segment(_fmt(leaf_loc), text, leaf_loc, KIND_RESULT, server, tool))


# --------------------------------------------------------------------------- results


def extract_result(method: str, result: Any, server: str, tool: str, *, spotlight: bool = True) -> Extraction:
    """Segments of a server → host result for ``method`` (tool = tool name, URI or prompt name)."""
    ex = Extraction()
    if not isinstance(result, dict):
        return ex
    if method == "tools/call":
        for i, block in enumerate(result.get("content") or []):
            _content_block(ex, block, ("content", i), server, tool, spotlight=spotlight)
        if "structuredContent" in result:
            _structured(ex, result["structuredContent"], ("structuredContent",), server, tool)
    elif method == "resources/read":
        for i, item in enumerate(result.get("contents") or []):
            _resource_contents(ex, item, ("contents", i), server, tool, spotlight=spotlight)
    elif method == "prompts/get":
        if isinstance(result.get("description"), str):
            ex.add(Segment("description", result["description"], ("description",), KIND_RESULT, server, tool))
        for i, msg in enumerate(result.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for j, block in enumerate(content):
                    _content_block(ex, block, ("messages", i, "content", j), server, tool, spotlight=spotlight)
            else:
                _content_block(ex, content, ("messages", i, "content"), server, tool, spotlight=spotlight)
    elif method == "completion/complete":
        completion = result.get("completion")
        values = completion.get("values") if isinstance(completion, dict) else None
        for i, value in enumerate(values or []):
            if isinstance(value, str):
                loc: Loc = ("completion", "values", i)
                ex.add(Segment(_fmt(loc), value, loc, KIND_RESULT, server, tool))
    elif method in ("initialize", "server/discover"):
        if isinstance(result.get("instructions"), str):
            ex.add(Segment("instructions", result["instructions"], ("instructions",), KIND_INSTRUCTIONS, server, tool))
    elif method == "resources/list":
        for i, item in enumerate(result.get("resources") or []):
            if isinstance(item, dict):
                for key in ("description", "title", "name"):
                    if isinstance(item.get(key), str) and len(item[key]) >= MIN_DESCRIPTION_LEN:
                        loc = ("resources", i, key)
                        ex.add(Segment(_fmt(loc), item[key], loc, KIND_DESCRIPTION, server, tool))
    else:
        # Unknown or extension method: every string is untrusted, nothing is rewritten.
        for leaf_loc, text in string_leaves(result):
            if leaf_loc and leaf_loc[0] in ("_meta", "resultType", "nextCursor"):
                continue
            ex.add(Segment(_fmt(leaf_loc), text, leaf_loc, KIND_RESULT, server, tool))
    return ex


# --------------------------------------------------------------------------- list items (pinning scan)


def extract_definition(kind: str, item: dict[str, Any], server: str) -> Extraction:
    """Description and schema strings of one tool / prompt / resource template."""
    ex = Extraction()
    name = str(item.get("name") or item.get("uriTemplate") or "")
    for key in ("description", "title"):
        if isinstance(item.get(key), str):
            ex.add(Segment(key, item[key], (key,), KIND_DESCRIPTION, server, name))
    annotations = item.get("annotations")
    if isinstance(annotations, dict) and isinstance(annotations.get("title"), str):
        ex.add(
            Segment("annotations.title", annotations["title"], ("annotations", "title"), KIND_DESCRIPTION, server, name)
        )
    if kind == "tool":
        for schema_key in ("inputSchema", "outputSchema"):
            _schema_strings(ex, item.get(schema_key), (schema_key,), server, name)
    elif kind == "prompt":
        for i, arg in enumerate(item.get("arguments") or []):
            if isinstance(arg, dict):
                for key in ("description", "title"):
                    if isinstance(arg.get(key), str):
                        loc: Loc = ("arguments", i, key)
                        ex.add(Segment(_fmt(loc), arg[key], loc, KIND_DESCRIPTION, server, name))
    # very short strings carry no instruction worth a classifier pass
    ex.segments = [s for s in ex.segments if len(s.text.strip()) >= MIN_DESCRIPTION_LEN]
    return ex


def _schema_strings(ex: Extraction, schema: Any, loc: Loc, server: str, tool: str) -> None:
    if isinstance(schema, dict):
        for key, value in schema.items():
            sub = (*loc, key)
            if key in _SCHEMA_TEXT_KEYS and isinstance(value, str):
                ex.add(Segment(_fmt(sub), value, sub, KIND_DESCRIPTION, server, tool))
            elif key in _SCHEMA_VALUE_KEYS:
                for leaf_loc, text in string_leaves(value, sub):
                    ex.add(Segment(_fmt(leaf_loc), text, leaf_loc, KIND_DESCRIPTION, server, tool))
            else:
                _schema_strings(ex, value, sub, server, tool)
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            _schema_strings(ex, value, (*loc, i), server, tool)


# --------------------------------------------------------------------------- requests and arguments


def extract_arguments(arguments: Any, server: str, tool: str, min_len: int, base: Loc = ("arguments",)) -> Extraction:
    """Free-text call arguments (string leaves at least ``min_len`` characters long)."""
    ex = Extraction()
    for loc, text in string_leaves(arguments, base):
        if len(text) >= min_len:
            ex.add(Segment(_fmt(loc), text, loc, KIND_ARGUMENT, server, tool))
    return ex


def extract_input_request(method: str, params: Any, server: str, loc: Loc) -> Extraction:
    """Server-authored prompts inside an MRTR input request or a legacy server request."""
    ex = Extraction()
    if not isinstance(params, dict):
        return ex
    tool = method
    if method == "elicitation/create":
        if isinstance(params.get("message"), str):
            sub = (*loc, "message")
            ex.add(Segment(_fmt(sub), params["message"], sub, KIND_INPUT_REQUEST, server, tool, spotlight=True))
        if isinstance(params.get("url"), str):
            sub = (*loc, "url")
            ex.add(Segment(_fmt(sub), params["url"], sub, KIND_INPUT_REQUEST, server, tool), second_pass=False)
        _schema_strings(ex, params.get("requestedSchema"), (*loc, "requestedSchema"), server, tool)
    elif method == "sampling/createMessage":
        if isinstance(params.get("systemPrompt"), str):
            sub = (*loc, "systemPrompt")
            ex.add(Segment(_fmt(sub), params["systemPrompt"], sub, KIND_INPUT_REQUEST, server, tool, spotlight=True))
        for i, msg in enumerate(params.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            blocks = content if isinstance(content, list) else [content]
            for j, block in enumerate(blocks):
                sub = (
                    (*loc, "messages", i, "content", j)
                    if isinstance(content, list)
                    else (*loc, "messages", i, "content")
                )
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    tsub = (*sub, "text")
                    ex.add(Segment(_fmt(tsub), block["text"], tsub, KIND_INPUT_REQUEST, server, tool, spotlight=True))
    elif method not in ("roots/list", "ping"):
        for leaf_loc, text in string_leaves(params, loc):
            ex.add(Segment(_fmt(leaf_loc), text, leaf_loc, KIND_INPUT_REQUEST, server, tool))
    return ex


def extract_notification(method: str, params: Any, server: str) -> Extraction:
    ex = Extraction()
    if not isinstance(params, dict):
        return ex
    if method == "notifications/progress":
        if isinstance(params.get("message"), str):
            ex.add(Segment("message", params["message"], ("message",), KIND_NOTIFICATION, server, method))
    elif method == "notifications/message":
        for loc, text in string_leaves(params.get("data"), ("data",)):
            ex.add(Segment(_fmt(loc), text, loc, KIND_NOTIFICATION, server, method))
    elif not method.endswith("/list_changed") and method not in ("notifications/resources/updated",):
        for loc, text in string_leaves(params):
            if loc and loc[0] == "_meta":
                continue
            ex.add(Segment(_fmt(loc), text, loc, KIND_NOTIFICATION, server, method))
    return ex


def extract_error(error: Any, server: str, tool: str) -> Extraction:
    ex = Extraction()
    if not isinstance(error, dict):
        return ex
    if isinstance(error.get("message"), str):
        ex.add(Segment("error.message", error["message"], ("message",), KIND_ERROR, server, tool))
    for loc, text in string_leaves(error.get("data"), ("data",)):
        ex.add(Segment(_fmt(("error", *loc)), text, loc, KIND_ERROR, server, tool))
    return ex


# --------------------------------------------------------------------------- location helpers


def get_at(obj: Any, loc: Loc) -> Any:
    for part in loc:
        obj = obj[part]
    return obj


def set_at(obj: Any, loc: Loc, value: Any) -> None:
    for part in loc[:-1]:
        obj = obj[part]
    obj[loc[-1]] = value
