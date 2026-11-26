"""Capability pinning + first-contact quarantine (02 §5).

* **Pin** — each tool / prompt / resource template's canonical JSON is hashed with
  SHA-256 per server. A changed or added item is withheld and raises
  ``SecurityTamperingError`` until the developer re-pins.
* **Scan** — description and schema strings go through Tier 3 at list time; a flagged
  item is withheld (``SecurityInjectionError``, class ``description_poisoning``).
* **Quarantine** — a server with no accepted pins exposes only ``read``-class tools and
  its results are handled under ``strict`` policy until a human accepts the manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .store import PinRow

STATUS_QUARANTINED = "quarantined"
STATUS_ACCEPTED = "accepted"
STATUS_PROBE_QUARANTINED = "probe_quarantined"

LIST_KEYS = {"tool": "tools", "prompt": "prompts", "template": "resourceTemplates"}
_DEF_KEYS = {
    "tool": ("name", "title", "description", "inputSchema", "outputSchema", "annotations", "execution"),
    "prompt": ("name", "title", "description", "arguments"),
    "template": ("name", "title", "description", "uriTemplate", "mimeType", "annotations"),
}

SCAN_ALLOW, SCAN_ABSTAIN, SCAN_BLOCK, SCAN_ERROR = "allow", "abstain", "block", "error"


def item_name(kind: str, item: dict[str, Any]) -> str:
    if kind == "template":
        return str(item.get("name") or item.get("uriTemplate") or "")
    return str(item.get("name") or "")


def canonical_def(kind: str, item: dict[str, Any]) -> dict[str, Any]:
    return {k: item[k] for k in _DEF_KEYS[kind] if k in item}


def def_hash(definition: dict[str, Any]) -> str:
    blob = json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Exposure:
    expose: bool
    reason: str | None = None
    event: str | None = None  # "tamper" | "injection" | "scan_error" | None


def exposure(
    server_status: str,
    kind: str,
    row: PinRow | None,
    *,
    current_hash: str | None = None,
    privilege: str | None = None,
    scan_decision: str | None = None,
) -> Exposure:
    """Is this item shown to the host (and callable)? Values default to the stored row."""
    if row is None and current_hash is None:
        return Exposure(False, "never listed by the server")
    current_hash = current_hash or (row.current_hash if row else None)
    privilege = privilege if privilege is not None else (row.privilege if row else None)
    scan_decision = scan_decision if scan_decision is not None else (row.scan_decision if row else None)
    accepted_hash = row.accepted_hash if row else None

    if scan_decision == SCAN_BLOCK:
        return Exposure(False, "description flagged by the classifier", "injection")
    if scan_decision in (SCAN_ERROR, None):
        return Exposure(False, "description not scanned (classifier unavailable)", "scan_error")
    if accepted_hash is not None and accepted_hash != current_hash:
        return Exposure(False, "definition changed since it was pinned", "tamper")
    if accepted_hash is None:
        if server_status == STATUS_ACCEPTED:
            return Exposure(False, "item added after the server was accepted", "tamper")
        if kind == "tool" and privilege != "read":
            return Exposure(False, f"server quarantined: {privilege} tools withheld until review")
        return Exposure(True)
    if server_status == STATUS_PROBE_QUARANTINED and kind == "tool" and privilege != "read":
        return Exposure(False, f"server quarantined by probe detection: {privilege} tools withheld")
    return Exposure(True)


def load_known_manifests() -> dict[str, set[str]]:
    """Published pins for reference servers: manifest name → set of item hashes."""
    try:
        text = resources.files("mcp_shield").joinpath("known_manifests.json").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    data = json.loads(text)
    return {name: set(hashes) for name, hashes in data.get("manifests", {}).items()}


def matches_known_manifest(hashes: set[str], known: dict[str, set[str]]) -> str | None:
    if not hashes:
        return None
    for name, pinned in known.items():
        if hashes <= pinned:
            return name
    return None


def diff_definitions(old: dict[str, Any] | None, new: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-level differences for the dashboard's pin viewer."""
    old = old or {}
    out = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            out.append({"field": key, "old": old.get(key), "new": new.get(key)})
    return out
