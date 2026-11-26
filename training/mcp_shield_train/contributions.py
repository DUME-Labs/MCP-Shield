"""Dataset contributions: one file per push, named after the contributor (see CONTRIBUTING-DATA.md).

Every team member — and anyone sending a pull request — adds files to ``contributions/``:

    contributions/<name>-<NNN>.jsonl      e.g.  rishikesh-001.jsonl, vineel-002.jsonl

``<name>`` is lower-case letters, digits and hyphens; ``<NNN>`` is that contributor's
push number, zero-padded, so the folder sorts by contributor and then in push order.
Each line is one segment::

    {"text": "...", "label": "indirect_injection", "source": "injecagent", "licence": "MIT",
     "server": "github", "tool": "get_issue", "field": "content[0].text",
     "group": "issue-2291", "hard_negative": false, "url": "...", "notes": "..."}

``check`` validates files before they are merged (exit code 1 on any error — suitable as a
pull-request check); ``manifest`` writes the ``dataset.toml`` that ``ingest`` turns into
the dataset, with one source per contribution file.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_shield.redact import redact
from mcp_shield.textnorm import fingerprint_text, normalize

from . import LABELS

FILENAME = re.compile(r"^(?P<member>[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?)-(?P<number>\d{3,})\.jsonl$")
REQUIRED = ("text", "label", "source", "licence")
OPTIONAL = (
    "server",
    "tool",
    "field",
    "context",
    "group",
    "hard_negative",
    "url",
    "notes",
    "collected_on",
    "language",
    "contains_fake_secret",
)
MIN_TEXT, MAX_TEXT = 8, 100_000


@dataclass
class FileReport:
    path: Path
    member: str | None = None
    number: int | None = None
    records: int = 0
    labels: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def text_hash(text: str) -> str:
    return hashlib.sha256(fingerprint_text(text).encode("utf-8")).hexdigest()


def check_file(path: Path, *, seen: dict[str, str] | None = None, heldout: set[str] | None = None) -> FileReport:
    """Validate one contribution file. ``seen`` maps text hashes to where they were first found."""
    report = FileReport(path)
    m = FILENAME.match(path.name)
    if not m:
        report.errors.append(f"file name must look like <name>-<NNN>.jsonl (e.g. rishikesh-001.jsonl), got {path.name}")
    else:
        report.member, report.number = m.group("member"), int(m.group("number"))
    seen = seen if seen is not None else {}
    labels: dict[str, int] = defaultdict(int)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        report.errors.append("file is not UTF-8")
        return report
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {n}"
        try:
            row = json.loads(line)
        except ValueError as exc:
            report.errors.append(f"{where}: not JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            report.errors.append(f"{where}: not a JSON object")
            continue
        missing = [k for k in REQUIRED if not str(row.get(k, "")).strip()]
        if missing:
            report.errors.append(f"{where}: missing {', '.join(missing)}")
            continue
        unknown = sorted(set(row) - set(REQUIRED) - set(OPTIONAL))
        if unknown:
            report.warnings.append(f"{where}: unknown fields {unknown} are kept but not used")
        if row["label"] not in LABELS:
            report.errors.append(f"{where}: label {row['label']!r} is not one of {', '.join(LABELS)}")
            continue
        text = str(row["text"])
        if not MIN_TEXT <= len(text) <= MAX_TEXT:
            report.errors.append(f"{where}: text must be {MIN_TEXT}–{MAX_TEXT} characters (got {len(text)})")
            continue
        if redact(normalize(text)) != normalize(text) and row.get("contains_fake_secret") is not True:
            report.errors.append(
                f"{where}: looks like it contains a real key, token or card number — remove it, or "
                f'mark an invented one with "contains_fake_secret": true'
            )
        if row.get("hard_negative") and row["label"] != "benign":
            report.errors.append(f"{where}: hard_negative is only meaningful for benign text")
        h = text_hash(text)
        if heldout and h in heldout:
            report.errors.append(f"{where}: this text is in a held-out benchmark and must not be used for training")
        elif h in seen:
            report.warnings.append(f"{where}: duplicate of {seen[h]} (it will be deduplicated)")
        else:
            seen[h] = f"{path.name}:{n}"
        labels[row["label"]] += 1
        report.records += 1
    if report.records == 0 and not report.errors:
        report.errors.append("file has no records")
    report.labels = dict(labels)
    return report


def check_folder(folder: Path, heldout: set[str] | None = None) -> list[FileReport]:
    seen: dict[str, str] = {}
    reports = [check_file(p, seen=seen, heldout=heldout) for p in sorted(folder.glob("*.jsonl"))]
    numbers: dict[str, list[int]] = defaultdict(list)
    for r in reports:
        if r.member is not None and r.number is not None:
            numbers[r.member].append(r.number)
    for member, nums in numbers.items():
        expected = list(range(1, max(nums) + 1))
        if sorted(nums) != expected:
            missing = sorted(set(expected) - set(nums))
            target = next(r for r in reports if r.member == member)
            target.warnings.append(f"{member}: push numbers are not consecutive (missing {missing})")
    others = [p.name for p in folder.iterdir() if p.is_file() and p.suffix != ".jsonl" and p.name != "README.md"]
    if others:
        reports.append(FileReport(folder, errors=[f"only <name>-<NNN>.jsonl files belong here: {others}"]))
    return reports


def heldout_hashes(paths: list[Path], text_field: str = "text") -> set[str]:
    hashes: set[str] = set()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get(text_field), str):
                    hashes.add(text_hash(row[text_field]))
    return hashes


def write_manifest(
    folder: Path,
    out: Path,
    *,
    name: str = "mcp-shield-clf",
    output: str = "data/assembled",
    heldout: list[tuple[str, Path, str]] | None = None,
) -> int:
    """A ``dataset.toml`` with one source per contribution file (sorted by contributor, then push)."""
    files = sorted(folder.glob("*.jsonl"), key=_order)
    lines = [
        "# Generated by `python -m mcp_shield_train contributions manifest` — regenerate rather than edit.",
        "[dataset]",
        f'name = "{name}"',
        f'output = "{output}"',
        "seed = 1234",
        "augment_copies = 1",
        "",
    ]
    for p in files:
        rel = Path(_relpath(p, out.parent)).as_posix()
        lines += [
            "[[sources]]",
            f'name = "{p.stem}"',
            f'path = "{rel}"',
            'text = "text"',
            'label = "label"',
            'source_column = "source"',
            'licence_column = "licence"',
            'group = "group"',
            'context = "context"',
            'server_column = "server"',
            'tool_column = "tool"',
            'field_column = "field"',
            'hard_negative_column = "hard_negative"',
            "",
        ]
    for hname, hpath, htext in heldout or []:
        lines += [
            "[[heldout]]",
            f'name = "{hname}"',
            f'path = "{Path(_relpath(hpath, out.parent)).as_posix()}"',
            f'text = "{htext}"',
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return len(files)


def _order(path: Path) -> tuple[str, int]:
    """Contributor, then push number as a number (so 1000 sorts after 999)."""
    m = FILENAME.match(path.name)
    return (m.group("member"), int(m.group("number"))) if m else (path.stem, 0)


def _relpath(path: Path, start: Path) -> str:
    import os

    try:
        return os.path.relpath(path.resolve(), start.resolve())
    except ValueError:  # different drives on Windows
        return str(path.resolve())


def summary(reports: list[FileReport]) -> dict[str, Any]:
    by_member: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in reports:
        for label, n in r.labels.items():
            by_member[r.member or "?"][label] += n
    return {
        "files": len([r for r in reports if r.member]),
        "records": sum(r.records for r in reports),
        "errors": sum(len(r.errors) for r in reports),
        "warnings": sum(len(r.warnings) for r in reports),
        "by_contributor": {m: dict(v) for m, v in sorted(by_member.items())},
    }
