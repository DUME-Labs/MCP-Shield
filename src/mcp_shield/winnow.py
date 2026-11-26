"""Winnowed fingerprints for the taint containment check (02 §4; Schleimer et al., 2003).

Text is canonicalised (:func:`mcp_shield.textnorm.fingerprint_text`), every character
``K``-gram is hashed, and in every window of ``W`` consecutive gram hashes the minimum
is kept. With ``K = 12`` and ``W = 21`` about one hash in eleven survives, and any
shared run of at least ``K + W - 1 = 32`` characters is guaranteed to share a kept
hash — wherever it sits in a long record.

The query side (a call's arguments) keeps *all* of its gram hashes, so the guarantee
holds against the record's winnowed set. A hit proves a shared 12-gram; when the
record's text is still in memory the shared run is then measured exactly, so short
common phrases do not count as a flow (see :mod:`mcp_shield.taint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from .textnorm import fingerprint_text

K = 12
W = 21
GUARANTEE = K + W - 1  # 32 characters

_BASE = np.uint64(0x100000001B3)  # FNV prime, odd
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_POWERS = np.array([pow(0x100000001B3, K - 1 - j, 1 << 64) for j in range(K)], dtype=np.uint64)


@dataclass(frozen=True)
class Fingerprints:
    """Winnowed hashes (sorted) and the text position of each one's gram.

    A hash appears once per kept position, so a gram that recurs in the text appears
    several times, each with its own position.
    """

    hashes: NDArray[Any]  # uint64, sorted ascending
    positions: NDArray[Any]  # int64, aligned with ``hashes``

    def __len__(self) -> int:
        return int(self.hashes.size)

    def to_bytes(self) -> bytes:
        return self.hashes.astype("<u8").tobytes() + self.positions.astype("<i8").tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> Fingerprints:
        n = len(data) // 16
        hashes = np.frombuffer(data[: n * 8], dtype="<u8").astype(np.uint64)
        positions = np.frombuffer(data[n * 8 : n * 16], dtype="<i8").astype(np.int64)
        return cls(hashes, positions)

    @classmethod
    def empty(cls) -> Fingerprints:
        return cls(np.empty(0, np.uint64), np.empty(0, np.int64))


def _codepoints(text: str) -> NDArray[Any]:
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4").astype(np.uint64)


def gram_hashes(canonical: str) -> NDArray[Any]:
    """Hash of every ``K``-gram of already-canonical text (uint64, position-aligned)."""
    if len(canonical) < K:
        return np.empty(0, np.uint64)
    cps = _codepoints(canonical)
    n = cps.size - K + 1
    with np.errstate(over="ignore"):
        h = np.zeros(n, dtype=np.uint64)
        for j in range(K):
            h += cps[j : j + n] * _POWERS[j]
        # splitmix64 finaliser: spreads the polynomial hash so window minima are uniform
        h ^= h >> np.uint64(30)
        h *= _MIX1
        h ^= h >> np.uint64(27)
        h *= _MIX2
        h ^= h >> np.uint64(31)
    return h


def winnow(canonical: str) -> Fingerprints:
    """Winnowed fingerprints of already-canonical text."""
    h = gram_hashes(canonical)
    if h.size == 0:
        return Fingerprints.empty()
    if h.size <= W:
        pos = np.array([int(np.argmin(h))], dtype=np.int64)
    else:
        windows = sliding_window_view(h, W)
        # rightmost minimum in each window (the paper's tie rule)
        rev = windows[:, ::-1]
        pos = np.arange(windows.shape[0], dtype=np.int64) + (W - 1 - np.argmin(rev, axis=1)).astype(np.int64)
        pos = np.unique(pos)
    hashes = h[pos]
    # Keep every kept position, even when two of them hold the same gram: the exact-run
    # check must be able to measure the run at the occurrence the query actually shares.
    order = np.argsort(hashes, kind="stable")
    return Fingerprints(hashes[order], pos[order])


def fingerprint(text: str) -> tuple[str, Fingerprints]:
    """Canonicalise ``text`` and winnow it; returns the canonical text too."""
    canonical = fingerprint_text(text)
    return canonical, winnow(canonical)


def find_hits(query_hashes: NDArray[Any], record: Fingerprints) -> list[tuple[int, int]]:
    """``(query_pos, record_pos)`` for every query gram and every kept position of its hash.

    Ordered by query position, so hits on one diagonal (``query_pos - record_pos``) arrive
    left to right.
    """
    if query_hashes.size == 0 or len(record) == 0:
        return []
    lo = np.searchsorted(record.hashes, query_hashes, side="left")
    hi = np.searchsorted(record.hashes, query_hashes, side="right")
    counts = hi - lo
    qpos = np.nonzero(counts)[0]
    if qpos.size == 0:
        return []
    reps = counts[qpos]
    total = int(reps.sum())
    first = np.repeat(lo[qpos], reps)
    offset = np.arange(total) - np.repeat(np.cumsum(reps) - reps, reps)
    rpos = record.positions[first + offset]
    return list(zip(np.repeat(qpos, reps).tolist(), rpos.tolist(), strict=True))


def _run_bounds(query: str, record: str, qpos: int, rpos: int) -> tuple[int, int]:
    """``(left, right)`` extent of the exact common run around a 12-gram hit; ``(0, 0)`` on a collision."""
    if query[qpos : qpos + K] != record[rpos : rpos + K]:
        return 0, 0  # hash collision
    left = 0
    while qpos - left - 1 >= 0 and rpos - left - 1 >= 0 and query[qpos - left - 1] == record[rpos - left - 1]:
        left += 1
    right = K
    while qpos + right < len(query) and rpos + right < len(record) and query[qpos + right] == record[rpos + right]:
        right += 1
    return left, right


def shared_run(query: str, record: str, qpos: int, rpos: int) -> int:
    """Length of the exact common run around a 12-gram hit (both texts canonical)."""
    left, right = _run_bounds(query, record, qpos, rpos)
    return left + right


def longest_shared_run(query: str, record_text: str | None, record: Fingerprints, min_run: int = GUARANTEE) -> int:
    """Longest verified shared run between ``query`` (canonical) and a record.

    Without the record's text (a record restored after an engine restart) any hit counts
    as a full-length run — the conservative reading of "any fingerprint hit".
    """
    hits = find_hits(gram_hashes(query), record)
    if not hits:
        return 0
    if record_text is None:
        return max(min_run, K)
    best = 0
    measured_to: dict[int, int] = {}  # diagonal -> query index the last measured run reached
    for qpos, rpos in hits:
        diag = qpos - rpos
        if qpos < measured_to.get(diag, -1):
            continue  # inside a run already measured on this diagonal
        left, right = _run_bounds(query, record_text, qpos, rpos)
        measured_to[diag] = qpos + right
        best = max(best, left + right)
        if best >= min_run:
            break
    return best
