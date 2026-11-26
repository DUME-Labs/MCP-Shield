"""Event recording: the local event store, the dashboard feed and the reference IDs."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import OrderedDict
from typing import Any

from ..redact import redact
from ..refid import new_ref
from ..store import Store

SEGMENT_STORE_LIMIT = 16 * 1024
RAW_CACHE_SIZE = 512


class EventBus:
    """Fan-out of decisions to dashboard WebSocket subscribers (bounded, drop-oldest)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)


class EventLog:
    def __init__(self, store: Store, bus: EventBus | None = None, clock: Any = time.time) -> None:
        self.store = store
        self.bus = bus or EventBus()
        self.clock = clock
        # raw (unredacted) segments for one-click actions; memory only, short-lived
        self._raw: OrderedDict[str, str] = OrderedDict()

    def emit(self, kind: str, verdict: str, *, segment: str | None = None, **fields: Any) -> str:
        ref = new_ref(self.store.ref_exists)
        event: dict[str, Any] = {"ref": ref, "ts": self.clock(), "kind": kind, "verdict": verdict, **fields}
        if segment is not None:
            self._raw[ref] = segment
            while len(self._raw) > RAW_CACHE_SIZE:
                self._raw.popitem(last=False)
            stored = redact(segment)
            if len(stored) > SEGMENT_STORE_LIMIT:
                stored = stored[:SEGMENT_STORE_LIMIT] + f"\n[… {len(stored) - SEGMENT_STORE_LIMIT} more characters]"
            event["segment"] = stored
        self.store.add_event(event)
        public = {k: v for k, v in event.items() if k != "segment"}
        public["has_segment"] = segment is not None
        self.bus.publish({"type": "event", "event": public})
        return ref

    def raw_segment(self, ref: str) -> str | None:
        return self._raw.get(ref)
