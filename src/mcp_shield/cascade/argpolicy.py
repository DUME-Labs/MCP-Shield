"""Tier 1 argument policy (02 §2, Tier 1 item 4).

Every string argument is scanned for shell metacharacters, path traversal and URL
smuggling. For fields typed as path, command or URL a hit is a **block**; for any other
string field a hit is an **annotation** that makes an ``execute`` / ``egress`` call run
under ``strict`` policy.

The operator set and the path canonicaliser are per-OS. In stdio mode the wrapped
server runs on this machine's OS, which decides; in HTTP mode the server may be remote
on a different OS, so both separator styles and both operator sets apply.
"""

from __future__ import annotations

import ipaddress
import ntpath
import os
import posixpath
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

from ..config import ToolPolicy

RULES_POSIX = "posix"
RULES_WINDOWS = "windows"
RULES_BOTH = "both"

FIELD_PATH = "path"
FIELD_COMMAND = "command"
FIELD_URL = "url"
FIELD_TEXT = "text"

_POSIX_OPS = (";", "|", "&&", "`", "$(", "${", "\n", "\r", "\x00")
_WINDOWS_OPS = (";", "|", "&", "^", "`", "$(", "\n", "\r", "\x00")
_WIN_ENV = re.compile(r"%[A-Za-z_][A-Za-z0-9_()]*%")
_SHORT_NAME = re.compile(r"~\d")
_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s\"'<>]+")
_TRAVERSAL_TEXT = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_PATH_NAMES = frozenset(
    {
        "path",
        "paths",
        "file",
        "files",
        "filename",
        "filenames",
        "filepath",
        "dir",
        "dirs",
        "directory",
        "directories",
        "folder",
        "cwd",
        "root",
        "dirname",
        "pathname",
        "workdir",
        "working_directory",
    }
)
_PATH_SUFFIXES = ("_path", "_paths", "_file", "_files", "_dir", "_dirs", "_directory", "_folder", "_filename")
_COMMAND_NAMES = frozenset(
    {
        "command",
        "commands",
        "cmd",
        "script",
        "shell",
        "command_line",
        "commandline",
        "executable",
        "program",
        "shell_command",
    }
)
_URL_NAMES = frozenset(
    {"url", "urls", "uri", "uris", "href", "link", "links", "endpoint", "webhook", "callback", "website", "homepage"}
)
_URL_SUFFIXES = ("_url", "_urls", "_uri", "_href", "_endpoint", "_link", "_webhook")
_URL_FORMATS = frozenset({"uri", "url", "iri", "uri-reference", "iri-reference"})


@dataclass(frozen=True)
class ArgFinding:
    path: str
    field_type: str
    kind: str  # metachar | traversal | outside_root | device_path | stream | url | nul
    detail: str
    block: bool


def host_rules() -> str:
    return RULES_WINDOWS if sys.platform == "win32" else RULES_POSIX


def snake(name: str) -> str:
    return _CAMEL.sub("_", name).replace("-", "_").lower()


def field_type_for(key: str, schema: dict[str, Any] | None, policy: ToolPolicy, dotted: str) -> str:
    """Type of one argument field: config, then schema ``format``, then its name."""
    for candidate in (dotted, key):
        if candidate in policy.field_types:
            return policy.field_types[candidate]
    if isinstance(schema, dict):
        fmt = schema.get("format")
        if not fmt and isinstance(schema.get("items"), dict):
            fmt = schema["items"].get("format")
        if isinstance(fmt, str) and fmt.lower() in _URL_FORMATS:
            return FIELD_URL
    name = snake(key)
    if name in _PATH_NAMES or name.endswith(_PATH_SUFFIXES) or name.startswith("path_"):
        return FIELD_PATH
    if name in _COMMAND_NAMES or name.endswith("_command") or name.endswith("_cmd"):
        return FIELD_COMMAND
    if name in _URL_NAMES or name.endswith(_URL_SUFFIXES):
        return FIELD_URL
    return FIELD_TEXT


# --------------------------------------------------------------------------- scanners


def metachar_hits(value: str, rules: str) -> list[str]:
    ops: set[str] = set()
    if rules in (RULES_POSIX, RULES_BOTH):
        ops.update(_POSIX_OPS)
    if rules in (RULES_WINDOWS, RULES_BOTH):
        ops.update(_WINDOWS_OPS)
    hits = sorted(op for op in ops if op in value)
    if rules in (RULES_WINDOWS, RULES_BOTH) and _WIN_ENV.search(value):
        hits.append("%VAR%")
    return [repr(h) if h in ("\n", "\r", "\x00") else h for h in hits]


def _separators(rules: str) -> str:
    return "/" if rules == RULES_POSIX else "/\\"


def _segments(value: str, rules: str) -> list[str]:
    return [s for s in re.split("[" + re.escape(_separators(rules)) + "]", value) if s != ""]


def has_traversal(value: str, rules: str) -> bool:
    return any(seg.strip() == ".." for candidate in {value, unquote(value)} for seg in _segments(candidate, rules))


def _is_windows_abs(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith(("\\\\", "//"))


def canonical_path(value: str, rules: str, *, local: bool) -> str:
    """Canonicalise a path for root comparison (never touches the filesystem when not local)."""
    path = value.strip()
    if path.startswith("~"):
        path = os.path.expanduser(path) if local else path
    windows = rules == RULES_WINDOWS or (rules == RULES_BOTH and (_is_windows_abs(path) or "\\" in path))
    if windows:
        if path.startswith(("\\\\?\\UNC\\", "//?/UNC/")):
            path = "\\\\" + path[8:]
        elif path.startswith(("\\\\?\\", "//?/")):
            path = path[4:]
        if local and sys.platform == "win32" and _SHORT_NAME.search(path):
            path = _long_path(path)
        if local and sys.platform == "win32" and os.path.exists(path):
            path = os.path.realpath(path)
        return ntpath.normcase(ntpath.normpath(path.replace("/", "\\")))
    if local and os.path.exists(path):
        path = os.path.realpath(path)
    return posixpath.normpath(path)


def _long_path(path: str) -> str:  # pragma: no cover - Windows only
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetLongPathNameW(path, buf, len(buf))  # type: ignore[attr-defined]
    return buf.value if 0 < n < len(buf) else path


def within(path: str, root: str, rules: str) -> bool:
    mod = (
        ntpath
        if (rules == RULES_WINDOWS or (rules == RULES_BOTH and ("\\" in root or _is_windows_abs(root))))
        else posixpath
    )
    try:
        return bool(mod.commonpath([path, root]) == root)
    except ValueError:  # different drives, or mixing absolute and relative
        return False


def check_path(value: str, rules: str, roots: list[str], *, local: bool) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    if "\x00" in value:
        problems.append(("nul", "NUL byte in path"))
    if has_traversal(value, rules):
        problems.append(("traversal", "'..' segment"))
    windows_rules = rules in (RULES_WINDOWS, RULES_BOTH)
    if windows_rules:
        bare = value.replace("/", "\\")
        if bare.startswith("\\\\.\\") or bare.startswith("\\\\?\\GLOBALROOT"):
            problems.append(("device_path", "Windows device path"))
        tail = bare[2:] if re.match(r"^[A-Za-z]:", bare) else bare
        if bare.startswith("\\\\?\\"):
            tail = re.sub(r"^\\\\\?\\(UNC\\)?([A-Za-z]:)?", "", bare)
        if ":" in tail:
            problems.append(("stream", "NTFS alternate data stream or stray ':'"))
    if roots:
        absolute = value.startswith(("/", "~")) or (
            windows_rules and (_is_windows_abs(value) or value.startswith("\\"))
        )
        if absolute:
            canon = canonical_path(value, rules, local=local)
            canon_roots = [canonical_path(os.path.expanduser(r), rules, local=local) for r in roots]
            if not any(within(canon, r, rules) for r in canon_roots):
                problems.append(("outside_root", "path outside the allowed roots"))
    elif windows_rules and value.replace("/", "\\").startswith("\\\\") and not value.startswith("\\\\?\\"):
        problems.append(("outside_root", "UNC path (network share)"))
    return problems


_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata", "instance-data"})


def check_url(value: str, policy: ToolPolicy) -> list[str]:
    problems: list[str] = []
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in value):
        problems.append("whitespace or control character in URL")
    if "\\" in value:
        problems.append("backslash in URL")
    try:
        parts = urlsplit(value)
        host = parts.hostname
        _ = parts.port
    except ValueError:
        return [*problems, "unparseable URL"]
    scheme = parts.scheme.lower()
    if scheme not in {s.lower() for s in policy.url_schemes}:
        problems.append(f"scheme '{scheme or '(none)'}' not allowed")
    if "@" in parts.netloc:
        problems.append("credentials or '@' in URL authority")
    if scheme in ("http", "https", "ws", "wss") and not host:
        problems.append("URL without a host")
    if host:
        if policy.url_hosts and not any(_host_match(host, pattern) for pattern in policy.url_hosts):
            problems.append(f"host '{host}' not on the allowlist")
        elif not policy.url_hosts and _link_local(host):
            problems.append(f"link-local / cloud-metadata address '{host}'")
    return problems


def _host_match(host: str, pattern: str) -> bool:
    host, pattern = host.lower().rstrip("."), pattern.lower().rstrip(".")
    if pattern.startswith("*."):
        return host.endswith(pattern[1:])
    return host == pattern


def _link_local(host: str) -> bool:
    if host.lower() in _METADATA_HOSTS:
        return True
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return bool(addr.is_link_local) or str(addr) == "fd00:ec2::254"


# --------------------------------------------------------------------------- the policy


def scan_arguments(
    arguments: Any,
    input_schema: dict[str, Any] | None,
    policy: ToolPolicy,
    rules: str,
    *,
    local: bool,
) -> list[ArgFinding]:
    """Scan every string argument; return findings (blocking or annotating)."""
    findings: list[ArgFinding] = []
    props = input_schema.get("properties") if isinstance(input_schema, dict) else None
    _walk(arguments, props if isinstance(props, dict) else {}, "", None, policy, rules, local, findings)
    return findings


def _walk(
    value: Any,
    props: dict[str, Any],
    dotted: str,
    ftype: str | None,
    policy: ToolPolicy,
    rules: str,
    local: bool,
    out: list[ArgFinding],
) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            key = str(key)
            schema = props.get(key) if isinstance(props.get(key), dict) else None
            sub_props = schema.get("properties") if schema and isinstance(schema.get("properties"), dict) else {}
            if schema and isinstance(schema.get("items"), dict) and isinstance(schema["items"].get("properties"), dict):
                sub_props = schema["items"]["properties"]
            sub_dotted = f"{dotted}.{key}" if dotted else key
            sub_type = field_type_for(key, schema, policy, sub_dotted)
            if sub_type == FIELD_TEXT and ftype and ftype != FIELD_TEXT and not isinstance(sub, (dict, list)):
                sub_type = ftype
            _walk(sub, sub_props or {}, sub_dotted, sub_type, policy, rules, local, out)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _walk(item, props, f"{dotted}[{i}]", ftype, policy, rules, local, out)
    elif isinstance(value, str):
        out.extend(_scan_string(value, dotted or "(root)", ftype or FIELD_TEXT, policy, rules, local))


def _scan_string(value: str, path: str, ftype: str, policy: ToolPolicy, rules: str, local: bool) -> list[ArgFinding]:
    found: list[ArgFinding] = []
    if ftype == FIELD_PATH:
        hits = metachar_hits(value, rules)
        if hits:
            found.append(ArgFinding(path, ftype, "metachar", " ".join(hits), True))
        for kind, detail in check_path(value, rules, policy.roots, local=local):
            found.append(ArgFinding(path, ftype, kind, detail, True))
    elif ftype == FIELD_COMMAND:
        hits = metachar_hits(value, rules)
        if hits and policy.mode != "shell":
            found.append(ArgFinding(path, ftype, "metachar", " ".join(hits), True))
        elif "\x00" in value:
            found.append(ArgFinding(path, ftype, "nul", "NUL byte", True))
    elif ftype == FIELD_URL:
        for detail in check_url(value, policy):
            found.append(ArgFinding(path, ftype, "url", detail, True))
        if "$(" in value or "`" in value:
            found.append(ArgFinding(path, ftype, "metachar", "shell substitution in URL", True))
    else:
        hits = metachar_hits(value, rules)
        if hits:
            found.append(ArgFinding(path, ftype, "metachar", " ".join(hits), False))
        if _TRAVERSAL_TEXT.search(value) or "..%2f" in value.lower() or "%2e%2e" in value.lower():
            found.append(ArgFinding(path, ftype, "traversal", "'..' path segment in text", False))
        for url in _URL_IN_TEXT.findall(value):
            if "@" in (urlsplit(url).netloc if _safe_split(url) else ""):
                found.append(ArgFinding(path, ftype, "url", "credentials or '@' in embedded URL", False))
    return found


def _safe_split(url: str) -> bool:
    try:
        urlsplit(url)
        return True
    except ValueError:
        return False
