"""Evaluation through the runtime classifier (03 §9).

Numbers are per **document** at the shipped operating point: FPR on benign documents,
recall on attack documents, by class and by source, plus the abstain rate. A model
directory is loaded with the engine's own :class:`OnnxClassifier`, so what is measured is
what the engine will do. ``loso_splits`` writes the leave-one-source-out datasets.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from mcp_shield.cascade.tier3_classifier import OnnxClassifier, SegmentInput

from .data import Record, write_jsonl


def evaluate(model_dir: Path, records: list[Record], fpr_target: float = 0.005) -> dict[str, Any]:
    clf = OnnxClassifier(model_dir, fpr_target=fpr_target)
    decisions: list[str] = []
    for start in range(0, len(records), 64):
        batch = records[start : start + 64]
        purpose = ["description" if "field=description" in r.context else "content" for r in batch]
        decisions += [
            v.decision
            for v in clf.classify([SegmentInput(r.context, r.text, p) for r, p in zip(batch, purpose, strict=True)])
        ]
    by_class: dict[str, list[str]] = defaultdict(list)
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r, d in zip(records, decisions, strict=True):
        by_class[r.label].append(d)
        by_source[r.source].append((r.label, d))

    def rate(ds: list[str], what: str) -> float | None:
        return sum(d == what for d in ds) / len(ds) if ds else None

    benign = by_class.get("benign", [])
    attacks = [d for lab, ds in by_class.items() if lab != "benign" for d in ds]
    return {
        "model": f"{clf.name} {clf.version}",
        "operating_point": clf.fpr_key,
        "documents": len(records),
        "fpr": rate(benign, "block"),
        "benign_abstain_rate": rate(benign, "abstain"),
        "recall": rate(attacks, "block"),
        "recall_by_class": {lab: rate(ds, "block") for lab, ds in by_class.items() if lab != "benign"},
        "by_source": {
            src: {
                "documents": len(items),
                "fpr": rate([d for lab, d in items if lab == "benign"], "block"),
                "recall": rate([d for lab, d in items if lab != "benign"], "block"),
            }
            for src, items in by_source.items()
        },
    }


def loso_splits(records: list[Record], out_dir: Path) -> dict[str, dict[str, int]]:
    """For each source: train/val on every other source, evaluate on the held-out one (03 §9.3)."""
    summary: dict[str, dict[str, int]] = {}
    for source in sorted({r.source for r in records}):
        held = [r for r in records if r.source == source]
        rest = [r for r in records if r.source != source]
        target = out_dir / f"loso-{source}"
        summary[source] = {
            "train": write_jsonl(target / "train.jsonl", [r for r in rest if r.split == "train"]),
            "val": write_jsonl(target / "val.jsonl", [r for r in rest if r.split == "val"]),
            "heldout": write_jsonl(target / "heldout.jsonl", held),
        }
    return summary
