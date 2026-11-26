"""Publish the accepted pins of a reviewed server as a known manifest.

    python scripts/pin_reference_servers.py --server filesystem --name server-filesystem@2026.7.0

Run a reference server once through MCP-Shield, review and accept its pins
(``mcp-shield pin --server <name> --accept``), then run this script: it copies the
accepted SHA-256 pins from the local store into ``src/mcp_shield/known_manifests.json``
under ``--name``. A user whose first listing of that server matches the manifest
exactly is spared the manual review step; anything else still starts quarantined.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from mcp_shield.store import Store  # noqa: E402

MANIFESTS = PROJECT / "src" / "mcp_shield" / "known_manifests.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="server name in the local pin store")
    p.add_argument("--name", required=True, help="manifest name, e.g. server-filesystem@2026.7.0")
    p.add_argument("--db", type=Path, help="pin store path (default: the per-user data directory)")
    args = p.parse_args()

    store = Store(args.db) if args.db else Store()
    try:
        pins = store.pins(args.server)
    finally:
        store.close()
    accepted = sorted(row.accepted_hash for row in pins.values() if row.accepted_hash and row.accepted)
    pending = [name for (_kind, name), row in pins.items() if not row.accepted]
    if not accepted:
        print(f"no accepted pins for {args.server!r}; review and accept them first", file=sys.stderr)
        return 1
    if pending:
        print(
            f"refusing: {len(pending)} definitions are not accepted ({', '.join(sorted(pending)[:5])})", file=sys.stderr
        )
        return 1
    data = json.loads(MANIFESTS.read_text(encoding="utf-8"))
    data.setdefault("manifests", {})[args.name] = accepted
    MANIFESTS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{args.name}: {len(accepted)} pins written to {MANIFESTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
