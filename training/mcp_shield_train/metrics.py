"""Operating-point metrics (03 §8, §9.1): everything is about the false-positive rate."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.floating[Any]]


def softmax(logits: Floats, temperature: float = 1.0) -> Floats:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: Floats = e / e.sum(axis=-1, keepdims=True)
    return out


def attack_prob(probs: Floats) -> Floats:
    """1 − p(benign); label 0 is benign."""
    out: Floats = 1.0 - np.asarray(probs)[..., 0]
    return out


def threshold_at_fpr(benign_scores: Floats, fpr: float) -> float:
    """Smallest threshold t with (#benign scores ≥ t) / n ≤ fpr."""
    s = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    n = s.size
    if n == 0:
        raise ValueError("no benign scores to set a threshold on")
    allowed = int(np.floor(fpr * n))
    if allowed >= n:
        return float(s[-1])
    # every score strictly above s[allowed] may be flagged; s[allowed] itself must not be
    return float(np.nextafter(s[allowed], np.inf))


def rate_at(scores: Floats, threshold: float) -> float:
    s = np.asarray(scores)
    return float((s >= threshold).mean()) if s.size else 0.0


def document_scores(chunk_scores: list[Floats]) -> Floats:
    """Max-pooled chunk scores, one per document (a document is flagged if any chunk is)."""
    return np.array([float(np.max(c)) for c in chunk_scores], dtype=np.float64)


def recall_at_fpr(benign: Floats, attacks: Floats, fpr: float) -> tuple[float, float]:
    t = threshold_at_fpr(benign, fpr)
    return rate_at(attacks, t), t


def expected_calibration_error(confidence: Floats, correct: NDArray[np.bool_], bins: int = 15) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(ok[mask].mean() - conf[mask].mean())
    return float(ece)


def nll(logits: Floats, labels: NDArray[np.integer[Any]], temperature: float) -> float:
    p = softmax(logits, temperature)
    idx = np.asarray(labels)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, 1.0))))
