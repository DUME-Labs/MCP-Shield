"""Dataset records (03 §3): one JSON object per line.

``{"id", "context", "text", "label", "source", "group_id", "split"?, "hard_negative"?}``

``context`` is the runtime prefix ``[server=… tool=… field=…]`` built by
:func:`mcp_shield.textnorm.context_prefix`; ``text`` is normalised exactly as the runtime
extractor normalises it, so training and inference see the same characters.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcp_shield.textnorm import normalize

from . import LABELS

_PREFIX = re.compile(r"^\[server=[A-Za-z0-9._:/@\-]+ tool=[A-Za-z0-9._:/@\-]+ field=[A-Za-z0-9._:/@\-]+\]$")
SPLITS = ("train", "val", "test_internal")


class DataError(ValueError):
    pass


@dataclass
class Record:
    id: str
    context: str
    text: str
    label: str
    source: str
    group_id: str
    split: str | None = None
    hard_negative: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        if d["split"] is None:
            d.pop("split")
        if not d["hard_negative"]:
            d.pop("hard_negative")
        return {**d, **extra}


def parse(obj: dict[str, Any], where: str = "") -> Record:
    missing = [k for k in ("id", "context", "text", "label", "source", "group_id") if not obj.get(k)]
    if missing:
        raise DataError(f"{where}: missing {', '.join(missing)}")
    if obj["label"] not in LABELS:
        raise DataError(f"{where}: unknown label {obj['label']!r}")
    if not _PREFIX.match(obj["context"]):
        raise DataError(f"{where}: context is not a runtime prefix: {obj['context'][:80]!r}")
    if obj.get("split") is not None and obj["split"] not in SPLITS:
        raise DataError(f"{where}: unknown split {obj['split']!r}")
    known = {"id", "context", "text", "label", "source", "group_id", "split", "hard_negative"}
    return Record(
        id=str(obj["id"]),
        context=obj["context"],
        text=normalize(str(obj["text"])),
        label=obj["label"],
        source=str(obj["source"]),
        group_id=str(obj["group_id"]),
        split=obj.get("split"),
        hard_negative=bool(obj.get("hard_negative", False)),
        extra={k: v for k, v in obj.items() if k not in known},
    )


def read_jsonl(path: Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                yield parse(json.loads(line), f"{path}:{n}")


def write_jsonl(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def check_unique_ids(records: list[Record]) -> None:
    seen: set[str] = set()
    for r in records:
        if r.id in seen:
            raise DataError(f"duplicate id {r.id}")
        seen.add(r.id)


def class_mix(records: list[Record]) -> dict[str, float]:
    total = max(1, len(records))
    mix = {label: sum(1 for r in records if r.label == label) / total for label in LABELS}
    benign = [r for r in records if r.label == "benign"]
    mix["hard_negative_share_of_benign"] = sum(r.hard_negative for r in benign) / max(1, len(benign))
    return mix
