"""ONNX export, quantisation and the shipped metadata (03 §1, §8).

1. Export the checkpoint with **two outputs** — ``logits`` and ``pooled`` (the ``[CLS]``
   vector used for near-duplicate rejection and taint embeddings) — and dynamic batch and
   sequence axes; write ``tokenizer.json`` so the runtime needs neither ``transformers``
   nor PyTorch.
2. Dynamic **int8** quantisation that covers the **embedding table** (``Gather``) as well
   as the ``MatMul`` weights — the 250k × 768 table is most of the model. An **fp16** graph
   is written for GPUs.
3. **Quantisation gate**: int8 ships only if it costs ≤ 0.5 points of recall at the
   operating FPR versus fp32 on validation; otherwise fp32 ships as ``model.onnx``.
4. **Calibrate on the shipped graph** through the runtime's own encoder, so thresholds
   match exactly what the engine will compute, then write ``mcp_shield_clf.json`` with
   labels, temperature, chunking, the threshold table, embedding calibration and the
   SHA-256 of every file.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from mcp_shield.cascade.tier3_classifier import META_FILE, OnnxClassifier, SegmentInput

from . import LABELS
from .calibrate import ValDoc, calibrate, embedding_calibration
from .data import Record
from .metrics import attack_prob, recall_at_fpr, softmax

QUANT_GATE = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_fp32(checkpoint: Path, out: Path, opset: int = 17) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    class WithPooled(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
            o = self.inner(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return o.logits, o.hidden_states[-1][:, 0]

    out.mkdir(parents=True, exist_ok=True)
    sample = tokenizer(["[server=a tool=b field=c]"], ["hello world"], return_tensors="pt")
    path = out / "model.fp32.onnx"
    kwargs: dict[str, Any] = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits", "pooled"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "pooled": {0: "batch"},
        },
        "opset_version": opset,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(WithPooled(model), (sample["input_ids"], sample["attention_mask"]), str(path), **kwargs)
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    return path


def quantize_int8(fp32: Path, out: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = out / "model.int8.onnx"
    quantize_dynamic(
        str(fp32), str(target), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather", "Gemm"]
    )
    return target


def convert_fp16(fp32: Path, out: Path) -> Path | None:
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError:
        return None
    model = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
    target = out / "model.fp16.onnx"
    onnx.save(model, str(target))
    return target


def _write_meta(directory: Path, meta: dict[str, Any]) -> None:
    (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def runtime_logits(directory: Path, records: list[Record]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Chunk logits and pooled vectors computed exactly as the engine computes them."""
    clf = OnnxClassifier(directory, fpr_target=0.005)
    logits: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for start in range(0, len(records), 64):
        batch = records[start : start + 64]
        per_seg = [clf.encode(SegmentInput(r.context, r.text)) for r in batch]
        flat = [e for encs in per_seg for e in encs]
        lg, pl = clf._run(flat)
        offset = 0
        for encs in per_seg:
            n = len(encs)
            logits.append(lg[offset : offset + n])
            if pl is not None:
                pooled.append(pl[offset : offset + n].mean(axis=0))
            offset += n
    return logits, pooled


def _doc_recall(logits: list[np.ndarray], records: list[Record], fpr: float) -> float:
    scores = np.array([float(np.max(attack_prob(softmax(c)))) for c in logits])
    benign = np.array([r.label == "benign" for r in records])
    if benign.all() or not benign.any():
        return 0.0
    return recall_at_fpr(scores[benign], scores[~benign], fpr)[0]


def build_release(
    checkpoint: Path,
    out: Path,
    val: list[Record],
    *,
    version: str,
    max_length: int,
    stride: int,
    training_info: dict[str, Any] | None = None,
    fp16: bool = True,
) -> dict[str, Any]:
    work = out / "_work"
    fp32 = export_fp32(checkpoint, work)
    int8 = quantize_int8(fp32, work)
    provisional = {
        "name": "mcp-shield-clf",
        "version": version,
        "labels": list(LABELS),
        "temperature": 1.0,
        "max_length": max_length,
        "stride": stride,
        "thresholds": {"0.005": {"block": {"1": 0.99}, "abstain": {"1": 0.98}}},
    }

    def staged(model: Path, name: str) -> Path:
        d = work / name
        d.mkdir(exist_ok=True)
        shutil.copy2(model, d / "model.onnx")
        shutil.copy2(work / "tokenizer.json", d / "tokenizer.json")
        _write_meta(d, provisional)
        return d

    fp32_logits, _ = runtime_logits(staged(fp32, "fp32"), val)
    int8_logits, _ = runtime_logits(staged(int8, "int8"), val)
    r32, r8 = _doc_recall(fp32_logits, val, 0.005), _doc_recall(int8_logits, val, 0.005)
    ship_int8 = r8 >= r32 - QUANT_GATE
    shipped = int8 if ship_int8 else fp32

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shipped, out / "model.onnx")
    shutil.copy2(work / "tokenizer.json", out / "tokenizer.json")
    if fp16:
        f16 = convert_fp16(fp32, work)
        if f16 is not None:
            shutil.copy2(f16, out / "model.fp16.onnx")
    _write_meta(out, provisional)

    logits, pooled = runtime_logits(out, val)
    docs = [
        ValDoc(
            lg,
            r.label,
            r.hard_negative,
            "description"
            if r.context.endswith("field=description]") or "field=inputSchema" in r.context
            else "content",
            r.group_id,
        )
        for lg, r in zip(logits, val, strict=True)
    ]
    cal = calibrate(docs)
    benign_idx = [i for i, r in enumerate(val) if r.label == "benign"]
    embedding = None
    if pooled:
        embedding = (
            embedding_calibration(np.stack([pooled[i] for i in benign_idx]), [val[i].group_id for i in benign_idx])
            if len(benign_idx) >= 50
            else None
        )
    meta: dict[str, Any] = {**provisional, **cal.to_meta()}
    if embedding:
        meta["embedding"] = embedding
    meta["files"] = {
        name: sha256(out / name)
        for name in ("model.onnx", "model.fp16.onnx", "tokenizer.json")
        if (out / name).exists()
    }
    meta["training"] = {
        **(training_info or {}),
        "quantization": {
            "shipped": "int8" if ship_int8 else "fp32",
            "recall_fp32": r32,
            "recall_int8": r8,
            "gate": QUANT_GATE,
        },
    }
    meta["calibration_report"] = cal.report
    _write_meta(out, meta)
    shutil.rmtree(work, ignore_errors=True)
    return meta
