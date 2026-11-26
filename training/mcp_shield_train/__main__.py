"""``python -m mcp_shield_train <command>`` — run from the ``training`` directory.

  ingest    --manifest dataset.toml               collected files → train / val / test_internal + card
  check     --data data/v1                        class mix, hard-negative share, size floor, leakage
  train     --data data/v1 --out runs/r1           fine-tune (add --base-model for a local copy)
  export    --run runs/r1 --data data/v1 --out models/v1 --version 0.1.0
  evaluate  --model models/v1 --data file.jsonl   per-document FPR / recall at the shipped threshold
  loso      --data data/v1 --out loso/            leave-one-source-out datasets
  contributions check contributions/              validate <name>-<NNN>.jsonl files (PR check)
  contributions manifest --folder contributions --out dataset.toml

Then install the result into MCP-Shield with ``mcp-shield pull-model --from models/v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import read_jsonl


def _load(data: Path, split: str) -> list:
    return list(read_jsonl(data / f"{split}.jsonl"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="mcp_shield_train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("--manifest", required=True, type=Path)
    s = sub.add_parser("check")
    s.add_argument("--data", required=True, type=Path)
    s = sub.add_parser("train")
    s.add_argument("--data", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--base-model", default="microsoft/mdeberta-v3-base")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--max-epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--grad-accum", type=int, default=2)
    s.add_argument("--lr", type=float, default=2e-5)
    s.add_argument("--max-length", type=int, default=512)
    s.add_argument("--stride", type=int, default=64, help="tokens of overlap between chunks of a long segment")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no-fp16", action="store_true")
    s.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)  # tiny random backbone, pipeline tests only
    s = sub.add_parser("export")
    s.add_argument("--run", required=True, type=Path)
    s.add_argument("--data", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--version", required=True)
    s.add_argument("--no-fp16", action="store_true")
    s = sub.add_parser("evaluate")
    s.add_argument("--model", required=True, type=Path)
    s.add_argument("--data", required=True, type=Path)
    s.add_argument("--fpr-target", type=float, default=0.005)
    s = sub.add_parser("contributions", help="check contribution files, or build a manifest from them")
    csub = s.add_subparsers(dest="action", required=True)
    c = csub.add_parser("check")
    c.add_argument("paths", nargs="+", type=Path, help="contribution files, or the contributions folder")
    c.add_argument("--heldout", nargs="*", type=Path, default=[], help="held-out JSONL files (text field 'text')")
    c = csub.add_parser("manifest")
    c.add_argument("--folder", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--name", default="mcp-shield-clf")
    c.add_argument("--output", default="data/assembled", help="dataset output folder, relative to the manifest")
    c.add_argument("--heldout", nargs="*", default=[], help="NAME=PATH[:TEXT_FIELD] entries")
    s = sub.add_parser("loso")
    s.add_argument("--data", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    from mcp_shield.cli import utf8_when_redirected

    utf8_when_redirected()
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        from .ingest import build, load_manifest

        card = build(load_manifest(args.manifest))
        print(json.dumps({k: card[k] for k in ("files", "split", "dedup", "removed_as_heldout")}, indent=2))
        for w in card["quality"]["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
        return 0
    if args.cmd == "check":
        from .ingest import quality_report
        from .split import leakage

        records = [r for split in ("train", "val", "test_internal") for r in _load(args.data, split)]
        report = quality_report([r for r in records if not r.extra.get("augmented_from")])
        report["split_leakage"] = leakage(records)
        print(json.dumps(report, indent=2))
        return 1 if report["split_leakage"] else 0
    if args.cmd == "train":
        from .train import TrainConfig, train

        cfg = TrainConfig(
            base_model=args.base_model,
            epochs=args.epochs,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            max_length=args.max_length,
            stride=args.stride,
            seed=args.seed,
            fp16=not args.no_fp16,
            smoke=args.smoke,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        summary = train(_load(args.data, "train"), _load(args.data, "val"), args.out, cfg)
        print(json.dumps({k: summary[k] for k in ("best_epoch", "best_val_recall_at_fpr", "device")}, indent=2))
        return 0
    if args.cmd == "export":
        from .export import build_release

        summary = json.loads((args.run / "train_summary.json").read_text(encoding="utf-8"))
        cfg = summary["config"]
        card_path = args.data / "dataset_card.json"
        info = {
            "base_model": cfg["base_model"],
            "seed": cfg["seed"],
            "best_epoch": summary["best_epoch"],
            "dataset": json.loads(card_path.read_text(encoding="utf-8"))["name"] if card_path.exists() else None,
        }
        meta = build_release(
            args.run / "checkpoint",
            args.out,
            _load(args.data, "val"),
            version=args.version,
            max_length=cfg["max_length"],
            stride=cfg["stride"],
            training_info=info,
            fp16=not args.no_fp16,
        )
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "temperature": meta["temperature"],
                    "quantization": meta["training"]["quantization"],
                    "recall": meta["calibration_report"],
                },
                indent=2,
            )
        )
        print(f"install with: mcp-shield pull-model --from {args.out}")
        return 0
    if args.cmd == "evaluate":
        from .evaluate import evaluate

        print(json.dumps(evaluate(args.model, list(read_jsonl(args.data)), args.fpr_target), indent=2))
        return 0
    if args.cmd == "contributions":
        from . import contributions as contrib

        if args.action == "check":
            held = contrib.heldout_hashes(args.heldout) if args.heldout else None
            if len(args.paths) == 1 and args.paths[0].is_dir():
                reports = contrib.check_folder(args.paths[0], held)
            else:
                seen: dict[str, str] = {}
                reports = [contrib.check_file(p, seen=seen, heldout=held) for p in args.paths]
            for r in reports:
                status = "ok" if r.ok else "FAILED"
                print(f"{r.path.name}: {status} — {r.records} records {r.labels}")
                for e in r.errors:
                    print(f"  error: {e}")
                for w in r.warnings:
                    print(f"  warning: {w}")
            print(json.dumps(contrib.summary(reports), indent=2))
            return 0 if all(r.ok for r in reports) else 1
        heldout = []
        for spec in args.heldout:
            hname, _, rest = spec.partition("=")
            hpath, _, field = rest.partition(":")
            heldout.append((hname, Path(hpath), field or "text"))
        n = contrib.write_manifest(args.folder, args.out, name=args.name, output=args.output, heldout=heldout)
        print(
            f"{args.out}: {n} contribution files; build with  python -m mcp_shield_train ingest --manifest {args.out}"
        )
        return 0
    if args.cmd == "loso":
        from .evaluate import loso_splits

        records = [r for split in ("train", "val", "test_internal") for r in _load(args.data, split)]
        print(json.dumps(loso_splits([r for r in records if not r.extra.get("augmented_from")], args.out), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
