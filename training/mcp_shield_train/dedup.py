"""Global deduplication *within* each label, then group merging (03 §5 step 1).

MinHash over character 5-gram shingles with LSH banding finds near-duplicates
(Jaccard ≳ threshold). Duplicates are removed only within a label: an attack and its
minimal benign twin are near-duplicates by construction and both must stay. Pairs that
are near-duplicates *across* labels are kept and merged into one ``group_id`` so a split
can never put them on opposite sides. An optional embedding pass (cosine > 0.92) can be
added with :func:`embedding_pairs` when vectors are available.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mcp_shield.textnorm import fingerprint_text

from .data import Record

NUM_PERM = 128
BANDS = 32  # 32 bands × 4 rows ⇒ ~50 % detection at Jaccard 0.42, ~99.7 % at 0.8
_PRIME = (1 << 32) + 15  # a prime just above 2^32: a·x stays below 2^64 for 32-bit a and x


def shingles(text: str, k: int = 5) -> set[int]:
    """32-bit hashes of the character k-grams of the canonical text."""
    t = fingerprint_text(text)
    grams = [t] if len(t) < k else [t[i : i + k] for i in range(len(t) - k + 1)]
    return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "little") for g in grams}


class MinHasher:
    """h_i(x) = (a_i·x + b_i) mod p, min over the set; exact in uint64 (no overflow)."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = 13) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 1 << 32, size=num_perm, dtype=np.uint64)[:, None]
        self.b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)[:, None]

    def signature(self, items: set[int]) -> NDArray[np.uint64]:
        x = np.fromiter(items, dtype=np.uint64)[None, :]
        p = np.uint64(_PRIME)
        h = ((self.a * x) % p + self.b) % p
        out: NDArray[np.uint64] = h.min(axis=1)
        return out


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def candidate_pairs(signatures: list[NDArray[np.uint64]], bands: int = BANDS) -> set[tuple[int, int]]:
    rows = len(signatures[0]) // bands if signatures else 0
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, sig in enumerate(signatures):
            buckets[sig[band * rows : (band + 1) * rows].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 500:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        pairs.add((members[x], members[y]))
    return pairs


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def deduplicate(
    records: list[Record],
    threshold: float = 0.8,
    group_threshold: float = 0.5,
    extra_pairs: Callable[[list[Record]], set[tuple[int, int]]] | None = None,
) -> tuple[list[Record], dict[str, int]]:
    """Drop same-label pairs at Jaccard ≥ ``threshold``; merge into one group every pair at ≥ ``group_threshold``.

    An attack and its benign twin differ by the whole injected instruction, so they are
    often only moderately similar (Jaccard 0.5–0.7); merging at the lower threshold keeps
    them — and near-similar siblings of one label — on the same side of the split.
    Merging only constrains the split; it never removes data.
    """
    hasher = MinHasher()
    sh = [shingles(r.text) for r in records]
    sigs = [hasher.signature(s) for s in sh]
    similar = {(i, j): jaccard(sh[i], sh[j]) for i, j in candidate_pairs(sigs)}
    near = {pair for pair, jac in similar.items() if jac >= min(threshold, group_threshold)}
    if extra_pairs is not None:
        extra = extra_pairs(records)
        near |= extra
        similar.update(dict.fromkeys(extra, 1.0))
    drop: set[int] = set()
    groups = UnionFind(len(records))
    # group_id ties siblings together already
    by_group: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.group_id in by_group:
            groups.union(by_group[r.group_id], i)
        else:
            by_group[r.group_id] = i
    within = across = 0
    for i, j in sorted(near):
        if records[i].label == records[j].label and similar[(i, j)] >= threshold:
            if i not in drop and j not in drop:
                drop.add(j)
                within += 1
        else:
            groups.union(i, j)
            across += records[i].label != records[j].label
    kept: list[Record] = []
    for i, r in enumerate(records):
        if i in drop:
            continue
        root = records[groups.find(i)].group_id
        if root != r.group_id:
            r.extra.setdefault("original_group_id", r.group_id)
            r.group_id = root
        kept.append(r)
    return kept, {
        "input": len(records),
        "dropped_within_label": within,
        "merged_across_labels": across,
        "kept": len(kept),
    }


def embedding_pairs(vectors: NDArray[np.floating], threshold: float = 0.92, block: int = 2048) -> set[tuple[int, int]]:
    """Index pairs whose (unit-normalised) vectors have cosine above ``threshold``."""
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    pairs: set[tuple[int, int]] = set()
    for start in range(0, len(v), block):
        sims = v[start : start + block] @ v.T
        for a, b in zip(*np.nonzero(sims > threshold), strict=True):
            i, j = start + int(a), int(b)
            if i < j:
                pairs.add((i, j))
    return pairs
