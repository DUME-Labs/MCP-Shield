"""Calibration and threshold selection (03 §8) → the metadata the runtime reads.

1. **Temperature scaling** on validation logits (NLL is convex in 1/T; ternary search).
2. **Block threshold** on the attack probability 1 − p(benign), chosen on validation
   **hard negatives** for FPR ≤ target (0.5 % by default); **abstain** threshold at ≈ 2 %.
3. **Document-level, length-adjusted**: a document is flagged if any chunk is, so the
   threshold for a document of *n* chunks is set so the *per-document* FPR stays at the
   target — empirically from long benign documents where there are enough, otherwise from
   the chunk-level benign distribution at per-chunk FPR 1 − (1 − F)^(1/n). Monotone in n.
4. **Per-class**: ``description_poisoning`` (list-time scans, where a false positive only
   hides a tool until reviewed) may use a looser FPR on benign descriptions, never a
   stricter threshold than the general one.
5. **Embedding calibration** for the taint near-paraphrase rule: the mean pooled vector
   (for centring) and a cosine threshold above what unrelated benign pairs reach; omitted
   when the vectors cannot separate them, which turns the rule off at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import Floats, attack_prob, nll, softmax, threshold_at_fpr

BUCKETS = (1, 2, 4, 8, 16, 32)
FPR_TARGETS = (0.005, 0.01, 0.02)
ABSTAIN_FPR = 0.02
MIN_EMPIRICAL_DOCS = 200


def fit_temperature(
    logits: Floats, labels: NDArray[np.integer[Any]], lo: float = 0.05, hi: float = 20.0, iters: int = 80
) -> float:
    """Temperature minimising validation NLL (search over the scale s = 1/T)."""
    a, b = 1.0 / hi, 1.0 / lo
    for _ in range(iters):
        m1, m2 = a + (b - a) / 3, b - (b - a) / 3
        if nll(logits, labels, 1.0 / m1) < nll(logits, labels, 1.0 / m2):
            b = m2
        else:
            a = m1
    return float(1.0 / ((a + b) / 2))


@dataclass
class ValDoc:
    """One validation document: its chunk logits, label and how it is used."""

    chunk_logits: Floats
    label: str
    hard_negative: bool = False
    purpose: str = "content"  # "content" | "description"
    group_id: str = ""


@dataclass
class Calibration:
    temperature: float
    thresholds: dict[str, dict[str, Any]]
    report: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] | None = None

    def to_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"temperature": self.temperature, "thresholds": self.thresholds}
        if self.embedding:
            meta["embedding"] = self.embedding
        return meta


def _bucket_table(benign_docs: list[Floats], chunk_scores: Floats, fpr: float) -> dict[str, float]:
    table: dict[str, float] = {}
    previous = 0.0
    for b in BUCKETS:
        members = [float(np.max(c)) for c in benign_docs if len(c) >= b]
        if len(members) >= MIN_EMPIRICAL_DOCS and b > 1:
            t = threshold_at_fpr(np.array(members), fpr)
        else:
            per_chunk = 1.0 - (1.0 - fpr) ** (1.0 / b)
            t = threshold_at_fpr(chunk_scores, per_chunk)
        t = min(max(t, previous), 1.0)  # longer documents never get a looser threshold
        table[str(b)] = round(t, 6)
        previous = t
    return table


def calibrate(
    docs: list[ValDoc],
    *,
    fpr_targets: tuple[float, ...] = FPR_TARGETS,
    abstain_fpr: float = ABSTAIN_FPR,
    description_fpr: float = 0.02,
) -> Calibration:
    if not docs:
        raise ValueError("no validation documents")
    logits_flat = np.concatenate([d.chunk_logits for d in docs])
    # temperature is fitted on the true class of every chunk (benign chunks of attack documents are rare in
    # MCP-shaped data; treating every chunk with its document label is the standard approximation)
    class_ids = np.concatenate([np.full(len(d.chunk_logits), _class_id(d.label)) for d in docs])
    temperature = fit_temperature(logits_flat, class_ids)

    def scores(d: ValDoc) -> Floats:
        return attack_prob(softmax(d.chunk_logits, temperature))

    content = [d for d in docs if d.purpose == "content"]
    negatives = [scores(d) for d in content if d.label == "benign" and d.hard_negative]
    if len(negatives) < 50:
        negatives = [scores(d) for d in content if d.label == "benign"]  # report says so
    chunk_neg = np.concatenate(negatives) if negatives else np.array([0.0])
    benign_desc = [float(np.max(scores(d))) for d in docs if d.purpose == "description" and d.label == "benign"]

    thresholds: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"temperature": temperature, "hard_negatives": len(negatives)}
    attacks = {
        lab: [float(np.max(scores(d))) for d in content if d.label == lab]
        for lab in ("indirect_injection", "command_injection")
    }
    poison = [float(np.max(scores(d))) for d in docs if d.label == "description_poisoning"]
    abstain = _bucket_table(negatives, chunk_neg, abstain_fpr)
    for fpr in fpr_targets:
        block = _bucket_table(negatives, chunk_neg, fpr)
        entry: dict[str, Any] = {"block": block, "abstain": {k: min(v, block[k]) for k, v in abstain.items()}}
        if len(benign_desc) >= 20:
            desc_t = min(threshold_at_fpr(np.array(benign_desc), max(description_fpr, fpr)), block["1"])
            entry["class_block"] = {"description_poisoning": {"1": round(desc_t, 6)}}
        thresholds[f"{fpr:g}"] = entry
        report[f"recall@{fpr:g}"] = {lab: _recall(v, block["1"]) for lab, v in attacks.items()}
        if poison:
            t = entry.get("class_block", {}).get("description_poisoning", {}).get("1", block["1"])
            report[f"recall@{fpr:g}"]["description_poisoning"] = _recall(poison, t)
    return Calibration(temperature, thresholds, report)


def _class_id(label: str) -> int:
    from . import LABEL_ID

    return LABEL_ID[label]


def _recall(values: list[float], threshold: float) -> float | None:
    return float(np.mean(np.array(values) >= threshold)) if values else None


def embedding_calibration(
    vectors: Floats,
    group_ids: list[str],
    *,
    pairs: int = 20000,
    seed: int = 5,
    floor: float = 0.9,
    margin: float = 0.02,
    ceiling: float = 0.99,
) -> dict[str, Any] | None:
    """Mean vector and a cosine threshold above unrelated benign pairs (99.9th percentile + margin)."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.shape[0] < 50:
        return None
    mean = v.mean(axis=0)
    c = v - mean
    c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(c), size=pairs)
    j = rng.integers(0, len(c), size=pairs)
    keep = np.array([group_ids[a] != group_ids[b] for a, b in zip(i, j, strict=True)])
    if keep.sum() < 100:
        return None
    cos = np.sum(c[i[keep]] * c[j[keep]], axis=1)
    threshold = max(floor, float(np.quantile(cos, 0.999)) + margin)
    if threshold > ceiling:
        return None  # vectors cannot tell unrelated benign texts apart: leave the rule off
    return {"cosine": round(threshold, 4), "mean": [round(float(x), 6) for x in mean]}
