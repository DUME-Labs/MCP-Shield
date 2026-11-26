"""Turning the team's collected data into the training dataset (03 §3–§6).

A manifest (``dataset.toml``, see ``dataset.example.toml``) lists every source file and
how to read it: which column holds the text, the label (or one label for the whole
file, or a mapping from the source's own labels), the group / document id, the context
prefix (a column, or server / tool / field values), hard-negative flags and the licence.

``build`` then runs the fixed order of 03 §5:

1. read and validate every source (labels, prefixes, normalisation as at run time);
2. **held-out guard** — any text that also appears in a held-out benchmark file
   (AgentDojo, MCPSecBench, the red-team set, …) is removed from the pool and reported;
3. global deduplication within each label, cross-label near-duplicates merged into one group;
4. grouped, stratified 80/10/10 split;
5. augmentation on train only;
6. ``train.jsonl`` / ``val.jsonl`` / ``test_internal.jsonl`` plus a dataset card with
   per-source counts, licences, the class mix against the 03 §6 targets and warnings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_shield.textnorm import context_prefix, fingerprint_text

from . import LABELS
from .augment import augment
from .data import DataError, Record, class_mix, parse, write_jsonl
from .dedup import deduplicate
from .split import grouped_split, leakage

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

TARGET_MIX = {"benign": 0.85, "indirect_injection": 0.08, "command_injection": 0.04, "description_poisoning": 0.03}
MIN_HARD_NEGATIVE_SHARE = 0.25
MIN_TRAINING_SEGMENTS = 20_000


@dataclass
class Source:
    name: str
    path: Path
    licence: str = "unknown"
    format: str | None = None
    text: str = "text"
    label: str | None = "label"
    fixed_label: str | None = None
    label_map: dict[str, str] = field(default_factory=dict)
    group: str | None = None
    context: str | None = None
    server: str = "unknown"
    tool: str = "unknown"
    field: str = "content[0].text"
    server_column: str | None = None
    tool_column: str | None = None
    field_column: str | None = None
    hard_negative: bool = False
    hard_negative_column: str | None = None
    source_column: str | None = None  # per-row origin (e.g. a corpus name or URL host) instead of the file name
    licence_column: str | None = None
    id_column: str | None = None
    note: str = ""


@dataclass
class Manifest:
    name: str
    output: Path
    sources: list[Source]
    heldout: list[Source]
    seed: int = 1234
    augment_copies: int = 1
    dedup_threshold: float = 0.8
    group_threshold: float = 0.5


def load_manifest(path: Path) -> Manifest:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    ds = raw.get("dataset", {})

    def source(spec: dict[str, Any]) -> Source:
        spec = dict(spec)
        if "name" not in spec or "path" not in spec:
            raise DataError(f"every source needs name and path: {spec}")
        p = Path(spec.pop("path"))
        known = set(Source.__dataclass_fields__) - {"path"}
        unknown = set(spec) - known
        if unknown:
            raise DataError(f"source {spec['name']}: unknown keys {sorted(unknown)}")
        src = Source(path=p if p.is_absolute() else base / p, **spec)
        for value in [src.fixed_label, *src.label_map.values()]:
            if value is not None and value not in LABELS:
                raise DataError(f"source {src.name}: label {value!r} is not one of {LABELS}")
        return src

    sources = [source(s) for s in raw.get("sources", [])]
    heldout = [source(s) for s in raw.get("heldout", [])]
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise DataError("source names must be unique")
    out = Path(ds.get("output", "dataset"))
    return Manifest(
        name=str(ds.get("name", "mcp-shield-dataset")),
        output=out if out.is_absolute() else base / out,
        sources=sources,
        heldout=heldout,
        seed=int(ds.get("seed", 1234)),
        augment_copies=int(ds.get("augment_copies", 1)),
        dedup_threshold=float(ds.get("dedup_threshold", 0.8)),
        group_threshold=float(ds.get("group_threshold", 0.5)),
    )


# --------------------------------------------------------------------------- reading


def _get(row: dict[str, Any], key: str | None) -> Any:
    """A column, or a dotted path into nested JSON (``messages.0.content``)."""
    if key is None:
        return None
    value: Any = row
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def read_rows(src: Source) -> list[dict[str, Any]]:
    fmt = (src.format or src.path.suffix.lstrip(".")).lower()
    if not src.path.exists():
        raise DataError(f"source {src.name}: file not found: {src.path}")
    if fmt == "jsonl":
        with open(src.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if fmt == "json":
        data = json.loads(src.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        return [r for r in data if isinstance(r, dict)]
    if fmt in ("csv", "tsv"):
        with open(src.path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if fmt == "tsv" else ","))
    if fmt == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise DataError(f"source {src.name}: reading parquet needs pandas + pyarrow") from exc
        return list(pd.read_parquet(src.path).to_dict(orient="records"))
    raise DataError(f"source {src.name}: unsupported format {fmt!r} (jsonl, json, csv, tsv, parquet)")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def to_records(src: Source) -> tuple[list[Record], Counter[str]]:
    records: list[Record] = []
    skipped: Counter[str] = Counter()
    for n, row in enumerate(read_rows(src)):
        text = _get(row, src.text)
        if not isinstance(text, str) or not text.strip():
            skipped["no text"] += 1
            continue
        if src.fixed_label:
            label = src.fixed_label
        else:
            raw_label = _get(row, src.label)
            label = src.label_map.get(str(raw_label), str(raw_label)) if raw_label is not None else ""
        if label not in LABELS:
            skipped[f"unmapped label {label!r}"] += 1
            continue
        context = _get(row, src.context) if src.context else None
        if not isinstance(context, str) or not context:
            context = context_prefix(
                str(_get(row, src.server_column) or src.server),
                str(_get(row, src.tool_column) or src.tool),
                str(_get(row, src.field_column) or src.field),
            )
        group = _get(row, src.group)
        if group is None:
            group = "h" + hashlib.sha256(fingerprint_text(text).encode()).hexdigest()[:16]
        hard = src.hard_negative or (
            src.hard_negative_column is not None and _truthy(_get(row, src.hard_negative_column))
        )
        rid = _get(row, src.id_column) if src.id_column else n
        try:
            records.append(
                parse(
                    {
                        "id": f"{src.name}-{rid}",
                        "context": context,
                        "text": text,
                        "label": label,
                        "source": str(_get(row, src.source_column) or src.name) if src.source_column else src.name,
                        "group_id": f"{src.name}:{group}",
                        "hard_negative": bool(hard and label == "benign"),
                        "licence": str(_get(row, src.licence_column) or src.licence)
                        if src.licence_column
                        else src.licence,
                        "origin": src.name,
                    },
                    f"{src.path}#{n}",
                )
            )
        except DataError as exc:
            skipped[str(exc).split(": ", 1)[-1][:60]] += 1
    return records, skipped


# --------------------------------------------------------------------------- checks and build


def heldout_hashes(sources: list[Source]) -> set[str]:
    hashes: set[str] = set()
    for src in sources:
        for row in read_rows(src):
            text = _get(row, src.text)
            if isinstance(text, str) and text.strip():
                hashes.add(hashlib.sha256(fingerprint_text(text).encode()).hexdigest())
    return hashes


def quality_report(records: list[Record]) -> dict[str, Any]:
    mix = class_mix(records)
    warnings: list[str] = []
    for label, target in TARGET_MIX.items():
        if abs(mix[label] - target) > max(0.02, target * 0.35):
            warnings.append(f"{label}: {mix[label]:.1%} of the pool, target ≈ {target:.0%} (03 §6)")
    if mix["hard_negative_share_of_benign"] < MIN_HARD_NEGATIVE_SHARE:
        warnings.append(
            f"hard negatives are {mix['hard_negative_share_of_benign']:.1%} of benign; need ≥ 25 % (03 §4 D)"
        )
    train = sum(1 for r in records if r.split in (None, "train"))
    if train < MIN_TRAINING_SEGMENTS:
        warnings.append(f"{train} training segments; the floor is {MIN_TRAINING_SEGMENTS:,} (03 §4)")
    for label in LABELS:
        if not any(r.label == label for r in records):
            warnings.append(f"no {label} examples at all")
    return {
        "total": len(records),
        "mix": {k: round(v, 4) for k, v in mix.items()},
        "warnings": warnings,
        "by_source": dict(Counter(r.source for r in records)),
        "by_label": dict(Counter(r.label for r in records)),
    }


def build(manifest: Manifest) -> dict[str, Any]:
    pool: list[Record] = []
    per_source: dict[str, Any] = {}
    for src in manifest.sources:
        recs, skipped = to_records(src)
        per_source[src.name] = {
            "path": str(src.path),
            "licence": src.licence,
            "read": len(recs),
            "skipped": dict(skipped),
            "labels": dict(Counter(r.label for r in recs)),
            "note": src.note,
        }
        pool.extend(recs)

    held = heldout_hashes(manifest.heldout)
    leaked = [r for r in pool if hashlib.sha256(fingerprint_text(r.text).encode()).hexdigest() in held]
    if leaked:
        leaked_ids = {r.id for r in leaked}
        pool = [r for r in pool if r.id not in leaked_ids]

    pool, dedup_stats = deduplicate(pool, threshold=manifest.dedup_threshold, group_threshold=manifest.group_threshold)
    split_counts = grouped_split(pool, seed=manifest.seed)
    bad = leakage(pool)
    if bad:
        raise DataError(f"split leakage across groups: {bad[:5]}")
    train = [r for r in pool if r.split == "train"]
    extra = augment(train, copies=manifest.augment_copies, seed=manifest.seed) if manifest.augment_copies else []

    out = manifest.output
    counts = {
        "train": write_jsonl(out / "train.jsonl", train + extra),
        "val": write_jsonl(out / "val.jsonl", [r for r in pool if r.split == "val"]),
        "test_internal": write_jsonl(out / "test_internal.jsonl", [r for r in pool if r.split == "test_internal"]),
    }
    card = {
        "name": manifest.name,
        "seed": manifest.seed,
        "sources": per_source,
        "heldout_files": [str(s.path) for s in manifest.heldout],
        "removed_as_heldout": len(leaked),
        "dedup": dedup_stats,
        "split": split_counts,
        "augmented_train_records": len(extra),
        "files": counts,
        "quality": quality_report(pool),
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return card
