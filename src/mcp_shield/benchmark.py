"""``mcp-shield benchmark`` — latency and replay suites that run on any machine (02 §2, 03 §9).

* ``latency`` — synthetic tool results of increasing length through the real cascade
  (Tier 1 on a call, Tier 2 cache, Tier 3 classifier, end-to-end inspector path):
  p50 / p95 per stage, the numbers behind the canonical latency table.
* ``replay`` — a JSONL file of labelled segments (``{"text", "label", "context"?}``)
  through the classifier at the shipped operating point: per-document FPR on benign
  rows and recall on attack rows.

The held-out agent suites (AgentDojo, MCPSecBench) need their own packages and data and
live in the repository's ``benchmarks/`` directory.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import SecurityConfig

WORDS = (
    "build",
    "release",
    "parser",
    "module",
    "request",
    "latency",
    "server",
    "client",
    "window",
    "report",
    "the",
    "a",
    "of",
    "and",
    "to",
    "in",
    "is",
    "for",
    "with",
    "on",
)


def _text(n_words: int, seed: int) -> str:
    return " ".join(WORDS[(i * 7 + seed) % len(WORDS)] for i in range(n_words)) + "."


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def latency(config: SecurityConfig, rounds: int = 20) -> dict[str, Any]:
    from .cascade.privilege import privilege_of
    from .cascade.tier1_policy import check_tool_call
    from .cascade.tier3_classifier import SegmentInput
    from .config import ToolPolicy
    from .engine.core import Connection, Inspector
    from .models import load_configured_classifier
    from .store import Store
    from .textnorm import context_prefix

    clf = load_configured_classifier(config)
    out: dict[str, Any] = {"classifier": getattr(clf, "reason", None) or f"{clf.name} {clf.version}"}
    tool = {"name": "read_file", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}}
    t1 = []
    for _ in range(rounds * 10):
        t0 = time.perf_counter()
        check_tool_call({"path": "docs/readme.md"}, tool, ToolPolicy(roots=["/work"]), "posix", local=False)
        privilege_of(tool, "read_file", ToolPolicy())
        t1.append((time.perf_counter() - t0) * 1000)
    out["tier1_ms"] = {"p50": statistics.median(t1), "p95": _pct(t1, 0.95)}
    if getattr(clf, "reason", None):
        out["tier3"] = "skipped: classifier unavailable"
        return out
    sizes = {
        "short (~40 words)": 40,
        "medium (~200 words)": 200,
        "long (~380 words, ≈512 tokens)": 380,
        "large (~3000 words)": 3000,
    }
    t3: dict[str, Any] = {}
    for label, n in sizes.items():
        times = []
        for r in range(rounds):
            seg = SegmentInput(context_prefix("bench", "read_file", "content[0].text"), _text(n, r))
            t0 = time.perf_counter()
            clf.classify([seg])
            times.append((time.perf_counter() - t0) * 1000)
        t3[label] = {"p50": statistics.median(times), "p95": _pct(times, 0.95)}
    out["tier3_ms"] = t3

    async def e2e() -> list[float]:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "bench.db")
            insp = Inspector(config, store, clf)
            conn = Connection("bench", "stdio", False, "bench-host", "posix", False)
            store.touch_server("bench", "stdio", False)
            store.upsert_pin(
                "bench", "tool", "read_file", "h", tool, "read", ("allow", "benign", 0.0, time.time()), accept=True
            )
            store.accept_pins("bench")
            times = []
            for r in range(rounds):
                req = {
                    "jsonrpc": "2.0",
                    "id": r,
                    "method": "tools/call",
                    "params": {"name": "read_file", "arguments": {"path": "a.md"}},
                }
                res = {"jsonrpc": "2.0", "id": r, "result": {"content": [{"type": "text", "text": _text(200, r + 99)}]}}
                t0 = time.perf_counter()
                await insp.from_host(conn, req)
                await insp.from_server(conn, res)
                times.append((time.perf_counter() - t0) * 1000)
            insp.close()
            store.close()
            return times

    times = asyncio.run(e2e())
    out["end_to_end_ms"] = {"p50": statistics.median(times), "p95": _pct(times, 0.95)}
    return out


def replay(config: SecurityConfig, data: Path) -> dict[str, Any]:
    from .cascade.tier3_classifier import SegmentInput
    from .models import load_configured_classifier

    clf = load_configured_classifier(config)
    if getattr(clf, "reason", None):
        raise RuntimeError(f"classifier unavailable: {clf.reason}")  # type: ignore[attr-defined]
    benign = attacks = fp = tp = 0
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        verdict = clf.classify(
            [SegmentInput(row.get("context", "[server=replay tool=replay field=text]"), row["text"])]
        )[0]
        flagged = verdict.decision == "block"
        if row["label"] == "benign":
            benign += 1
            fp += flagged
        else:
            attacks += 1
            tp += flagged
    return {
        "benign_documents": benign,
        "fpr": fp / benign if benign else None,
        "attack_documents": attacks,
        "recall": tp / attacks if attacks else None,
    }
