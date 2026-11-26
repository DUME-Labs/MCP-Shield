"""Tier 2 — Cache & Signature Layer (02 §2). Hashing and exact matching only, 1–5 ms.

Tier 2 never decides that *new* untrusted text is safe:

* **fast-accept** — an exact SHA-256 hit on content Tier 3 already cleared (below the
  abstain band), scoped by server + tool, with a TTL; never a similarity match;
* **fast-reject** — exact / normalised-hash matches against the local signature set,
  plus containment of a confirmed attack string (its winnowed fingerprints);
  near-duplicate rejection by embedding runs right after Tier 3;
* **Context-DoS guard** — per-tool length caps (truncate-and-flag for read-class tools)
  and a compression-ratio check against padding and recursive-text payloads.
"""

from __future__ import annotations

import hashlib
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..textnorm import fingerprint_text
from ..winnow import Fingerprints, find_hits, gram_hashes, winnow

CacheKey = tuple[str, str, str]


class ClearedCache:
    """LRU of content cleared by Tier 3 (or allowed by the user), touched only from the event loop."""

    def __init__(self, max_entries: int = 50_000) -> None:
        self._entries: OrderedDict[CacheKey, float] = OrderedDict()
        self._max = max_entries

    def __len__(self) -> int:
        return len(self._entries)

    def hit(self, server: str, tool: str, sha256: str, now: float | None = None) -> bool:
        key = (server, tool, sha256)
        expires = self._entries.get(key)
        if expires is None:
            return False
        if expires < (now if now is not None else time.time()):
            del self._entries[key]
            return False
        self._entries.move_to_end(key)
        return True

    def put(self, server: str, tool: str, sha256: str, ttl_s: float, now: float | None = None) -> None:
        key = (server, tool, sha256)
        self._entries[key] = (now if now is not None else time.time()) + ttl_s
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def invalidate_server(self, server: str) -> None:
        for key in [k for k in self._entries if k[0] == server]:
            del self._entries[key]

    def remove(self, server: str, tool: str, sha256: str) -> None:
        self._entries.pop((server, tool, sha256), None)


def normalized_sha(text: str) -> str:
    return hashlib.sha256(fingerprint_text(text).encode("utf-8")).hexdigest()


@dataclass
class Signature:
    sha256: str
    norm_sha256: str
    fingerprints: Fingerprints
    canonical: str
    embedding: NDArray[Any] | None = None


@dataclass
class SignatureSet:
    min_containment_len: int = 32
    near_dup_cosine: float = 0.95
    _exact: dict[str, Signature] = field(default_factory=dict)
    _norm: dict[str, Signature] = field(default_factory=dict)
    _contain: list[Signature] = field(default_factory=list)
    _emb: list[NDArray[Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._exact)

    def add(self, text: str, embedding: NDArray[Any] | None = None) -> Signature:
        canonical = fingerprint_text(text)
        sig = Signature(
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            winnow(canonical),
            canonical,
            None if embedding is None else _unit(embedding),
        )
        self._exact[sig.sha256] = sig
        self._norm[sig.norm_sha256] = sig
        if len(canonical) >= self.min_containment_len:
            self._contain.append(sig)
        if sig.embedding is not None:
            self._emb.append(sig.embedding)
        return sig

    def match(self, text: str, sha256: str | None = None) -> str | None:
        """Why ``text`` is a known attack, or ``None``."""
        if sha256 and sha256 in self._exact:
            return "exact signature match"
        canonical = fingerprint_text(text)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() in self._norm:
            return "normalised signature match"
        if self._contain and len(canonical) >= self.min_containment_len:
            grams = gram_hashes(canonical)
            for sig in self._contain:
                for qpos, rpos in find_hits(grams, sig.fingerprints):
                    if _covers(canonical, sig.canonical, qpos, rpos):
                        return "contains a known attack string"
        return None

    def near_duplicate(self, embedding: NDArray[Any] | None) -> bool:
        if embedding is None or not self._emb:
            return False
        v = _unit(embedding)
        return bool(np.max(np.stack(self._emb) @ v) >= self.near_dup_cosine)


def _covers(query: str, sig: str, qpos: int, rpos: int) -> bool:
    start = qpos - rpos
    return start >= 0 and query[start : start + len(sig)] == sig


def _unit(v: NDArray[Any]) -> NDArray[Any]:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- Context-DoS guard


@dataclass(frozen=True)
class DosVerdict:
    action: str  # "ok" | "truncate" | "reject"
    reason: str = ""
    cap: int = 0


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(raw) / max(1, len(zlib.compress(raw, 1)))


def dos_check(
    total_bytes: int, joined_text: str, *, cap: int, read_class: bool, min_size: int, max_ratio: float
) -> DosVerdict:
    if total_bytes > cap:
        if read_class:
            return DosVerdict("truncate", f"result of {total_bytes} bytes exceeds the {cap}-byte cap", cap)
        return DosVerdict("reject", f"result of {total_bytes} bytes exceeds the {cap}-byte cap", cap)
    if total_bytes >= min_size:
        ratio = compression_ratio(joined_text)
        if ratio > max_ratio:
            return DosVerdict("reject", f"repetitive payload (compression ratio {ratio:.0f}:1)", cap)
    return DosVerdict("ok", cap=cap)
