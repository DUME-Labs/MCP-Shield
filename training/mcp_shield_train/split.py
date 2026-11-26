"""Grouped, stratified 80/10/10 split (03 §5 step 2).

Every segment derived from one document or conversation shares a ``group_id`` and lands
on one side of the split; groups are assigned greedily so each (label, source) stratum
approaches the target ratios. Deterministic for a given seed.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from .data import SPLITS, Record

RATIOS = {"train": 0.8, "val": 0.1, "test_internal": 0.1}


def grouped_split(
    records: list[Record], seed: int = 1234, ratios: dict[str, float] | None = None, respect_existing: bool = True
) -> dict[str, int]:
    ratios = ratios or RATIOS
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.group_id].append(r)

    def stratum(members: list[Record]) -> tuple[str, str]:
        label = Counter(m.label for m in members).most_common(1)[0][0]
        attack = [m for m in members if m.label != "benign"]
        return (attack[0].label if attack else label, members[0].source)

    totals: dict[tuple[str, str], int] = Counter()
    for members in groups.values():
        totals[stratum(members)] += len(members)
    assigned: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    order = list(groups.items())
    random.Random(seed).shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))  # big groups first, so small ones can even out the ratios
    for _gid, members in order:
        fixed = {m.split for m in members if m.split} if respect_existing else set()
        if len(fixed) > 1:
            raise ValueError(f"group {members[0].group_id} already spans splits {sorted(fixed)}")
        st = stratum(members)
        if fixed:
            split = fixed.pop()
        else:
            total = totals[st]

            def deficit(name: str, st: tuple[str, str] = st, total: int = total) -> float:
                return ratios[name] * total - assigned[st][name]

            split = max(SPLITS, key=deficit)
        for m in members:
            m.split = split
        assigned[st][split] += len(members)
    return dict(Counter(r.split for r in records if r.split))


def leakage(records: list[Record]) -> list[str]:
    """Group ids found in more than one split (must be empty)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.split:
            seen[r.group_id].add(r.split)
    return sorted(g for g, s in seen.items() if len(s) > 1)
