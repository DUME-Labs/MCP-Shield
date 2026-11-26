"""Tier 3 — Encoder Injection Classifier (02 §2, 03 §1/§8).

The model directory holds:

* ``model.onnx`` (CPU, int8) and optionally ``model.fp16.onnx`` (GPU) — two outputs:
  ``logits`` ``[batch, 4]`` and ``pooled`` ``[batch, hidden]`` (the ``[CLS]`` vector
  reused for near-duplicate rejection and taint embeddings);
* ``tokenizer.json`` — Hugging Face ``tokenizers``, so neither ``transformers`` nor
  PyTorch is needed at run time;
* ``mcp_shield_clf.json`` — labels, temperature, chunking and the threshold table.

Each segment is encoded as the pair ``(context prefix, text)``; long text overflows into
≤ ``max_length``-token chunks with ``stride`` tokens of overlap, every chunk carrying the
prefix. A document's score is the **max** over its chunks of the temperature-scaled
attack probability 1 − p(benign), compared with a **length-adjusted** threshold.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

LABELS = ("benign", "indirect_injection", "command_injection", "description_poisoning")
BENIGN = "benign"
META_FILE = "mcp_shield_clf.json"
DEFAULT_FPR_KEY = "0.005"

VERDICT_ALLOW = "allow"
VERDICT_ABSTAIN = "abstain"
VERDICT_BLOCK = "block"


class ClassifierUnavailable(RuntimeError):
    """Model missing, corrupt, or inference failed — the caller fails closed."""


@dataclass
class SegmentInput:
    prefix: str
    text: str
    purpose: str = "content"  # "content" | "description" | "argument"


@dataclass
class Verdict:
    decision: str  # allow | abstain | block
    label: str
    attack_prob: float
    probs: dict[str, float]
    chunks: int
    embedding: NDArray[Any] | None = None
    chunk_embeddings: list[NDArray[Any]] = field(default_factory=list)


class Classifier(Protocol):
    name: str
    version: str
    embedding_cosine: float | None

    def classify(self, segments: list[SegmentInput]) -> list[Verdict]: ...


# --------------------------------------------------------------------------- thresholds


@dataclass
class ThresholdTable:
    """Block and abstain thresholds by chunk count (keys are the smallest chunk count of a bucket)."""

    block: dict[int, float]
    abstain: dict[int, float]
    class_block: dict[str, dict[int, float]] = field(default_factory=dict)

    @staticmethod
    def _pick(table: dict[int, float], chunks: int) -> float:
        keys = sorted(table)
        chosen = keys[0]
        for k in keys:
            if k <= chunks:
                chosen = k
        return table[chosen]

    def block_threshold(self, chunks: int, label: str | None = None, purpose: str = "content") -> float:
        """The description-poisoning threshold applies to list-time scans only (03 §8 item 5);
        any other per-class threshold applies wherever that class is the verdict."""
        if purpose == "description" and "description_poisoning" in self.class_block:
            return self._pick(self.class_block["description_poisoning"], chunks)
        if label and label != "description_poisoning" and label in self.class_block:
            return self._pick(self.class_block[label], chunks)
        return self._pick(self.block, chunks)

    def abstain_threshold(self, chunks: int) -> float:
        return self._pick(self.abstain, chunks)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ThresholdTable:
        def conv(t: Any) -> dict[int, float]:
            if isinstance(t, (int, float)):
                return {1: float(t)}
            return {int(k): float(v) for k, v in dict(t).items()}

        return cls(
            block=conv(data["block"]),
            abstain=conv(data["abstain"]),
            class_block={str(k): conv(v) for k, v in dict(data.get("class_block", {})).items()},
        )


@dataclass
class ModelMeta:
    labels: tuple[str, ...]
    temperature: float
    max_length: int
    stride: int
    thresholds: dict[str, ThresholdTable]
    version: str = "unknown"
    name: str = "mcp-shield-clf"
    files: dict[str, str] = field(default_factory=dict)
    # Calibrated near-paraphrase threshold for the pooled vector, and the mean vector to
    # centre it with (fine-tuned [CLS] vectors cluster by class; uncentred, unrelated
    # benign texts can exceed any fixed cosine). Absent → the embedding rule is off.
    embedding_cosine: float | None = None
    embedding_mean: list[float] | None = None
    # A model trained without MCP-Shield's context prefix (an off-the-shelf classifier) gets the text alone.
    use_prefix: bool = True
    # Set for a stand-in model used until mcp-shield-clf is trained; doctor and the dashboard say so.
    interim: bool = False

    @classmethod
    def load(cls, path: Path) -> ModelMeta:
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = tuple(data.get("labels", LABELS))
        if labels[0] != BENIGN:
            raise ClassifierUnavailable(f"{path}: label 0 must be 'benign', got {labels[0]!r}")
        return cls(
            labels=labels,
            temperature=float(data.get("temperature", 1.0)),
            max_length=int(data.get("max_length", 512)),
            stride=int(data.get("stride", 64)),
            thresholds={str(k): ThresholdTable.from_json(v) for k, v in data["thresholds"].items()},
            version=str(data.get("version", "unknown")),
            name=str(data.get("name", "mcp-shield-clf")),
            files={str(k): str(v) for k, v in dict(data.get("files", {})).items()},
            embedding_cosine=float(emb["cosine"])
            if isinstance(emb := data.get("embedding"), dict) and "cosine" in emb
            else None,
            embedding_mean=[float(x) for x in emb["mean"]] if isinstance(emb, dict) and "mean" in emb else None,
            use_prefix=bool(data.get("use_prefix", True)),
            interim=bool(data.get("interim", False)),
        )

    def table_for(self, fpr_target: float) -> tuple[ThresholdTable, str]:
        keys = sorted(self.thresholds, key=lambda k: abs(float(k) - fpr_target))
        return self.thresholds[keys[0]], keys[0]


def softmax(logits: NDArray[Any], temperature: float) -> NDArray[Any]:
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: NDArray[Any] = e / e.sum(axis=-1, keepdims=True)
    return out


def decide(
    chunk_probs: NDArray[Any], labels: tuple[str, ...], table: ThresholdTable, purpose: str
) -> tuple[str, str, float, dict[str, float]]:
    """Document decision from per-chunk probabilities (max-pooled attack probability)."""
    attack = 1.0 - chunk_probs[:, 0]
    worst = int(np.argmax(attack))
    p = chunk_probs[worst]
    score = float(attack[worst])
    attack_label = labels[1 + int(np.argmax(p[1:]))] if len(labels) > 1 else BENIGN
    n = int(chunk_probs.shape[0])
    probs = {labels[i]: float(p[i]) for i in range(len(labels))}
    if score >= table.block_threshold(n, attack_label, purpose):
        return VERDICT_BLOCK, attack_label, score, probs
    if score >= table.abstain_threshold(n):
        return VERDICT_ABSTAIN, attack_label, score, probs
    return VERDICT_ALLOW, BENIGN, score, probs


# --------------------------------------------------------------------------- ONNX implementation


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


class OnnxClassifier:
    """ONNX Runtime classifier. Thread-safe: ``InferenceSession.run`` may be called concurrently."""

    def __init__(
        self,
        model_dir: Path,
        *,
        fpr_target: float = 0.005,
        providers: list[str] | None = None,
        prefer_fp16: bool = False,
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / META_FILE
        if not meta_path.exists():
            raise ClassifierUnavailable(f"model metadata missing: {meta_path}")
        self.meta = ModelMeta.load(meta_path)
        self.name, self.version = self.meta.name, self.meta.version
        self.embedding_cosine = self.meta.embedding_cosine
        self._emb_mean = (
            np.asarray(self.meta.embedding_mean, dtype=np.float32) if self.meta.embedding_mean is not None else None
        )
        self.table, self.fpr_key = self.meta.table_for(fpr_target)
        if abs(float(self.fpr_key) - fpr_target) > 1e-9:
            log.warning("fpr_target %s not shipped with the model; using %s", fpr_target, self.fpr_key)

        providers = providers or ["CPUExecutionProvider"]
        gpu = any(p != "CPUExecutionProvider" for p in providers)
        candidates = ["model.fp16.onnx", "model.onnx"] if (gpu or prefer_fp16) else ["model.onnx", "model.fp16.onnx"]
        model_path = next((self.model_dir / c for c in candidates if (self.model_dir / c).exists()), None)
        if model_path is None:
            raise ClassifierUnavailable(f"no model.onnx in {self.model_dir}")
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise ClassifierUnavailable(f"tokenizer.json missing in {self.model_dir}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        except Exception as exc:  # corrupt file, unsupported op, missing provider libraries
            raise ClassifierUnavailable(f"cannot load {model_path.name}: {exc}") from exc
        self.model_file = model_path.name
        self.providers = self.session.get_providers()
        self.input_names = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        if "logits" not in outputs:
            raise ClassifierUnavailable(f"{model_path.name} has no 'logits' output (outputs: {outputs})")
        self._outputs = ["logits"] + (["pooled"] if "pooled" in outputs else [])

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(
            max_length=self.meta.max_length,
            stride=self.meta.stride,
            strategy="only_second" if self.meta.use_prefix else "longest_first",
        )
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.pad_id = int(pad_id) if pad_id is not None else 0
        self._lock = threading.Lock()
        self.batch_size = 16

    def encode(self, seg: SegmentInput) -> list[Any]:
        enc = self.tokenizer.encode(seg.prefix, seg.text) if self.meta.use_prefix else self.tokenizer.encode(seg.text)
        return [enc, *enc.overflowing]

    def _run(self, encodings: list[Any]) -> tuple[NDArray[Any], NDArray[Any] | None]:
        logits_all: list[NDArray[Any]] = []
        pooled_all: list[NDArray[Any]] = []
        for start in range(0, len(encodings), self.batch_size):
            batch = encodings[start : start + self.batch_size]
            width = max(len(e.ids) for e in batch)
            ids = np.full((len(batch), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            types = np.zeros((len(batch), width), dtype=np.int64)
            for i, e in enumerate(batch):
                n = len(e.ids)
                ids[i, :n] = e.ids
                mask[i, :n] = e.attention_mask
                types[i, :n] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            feed = {k: v for k, v in feed.items() if k in self.input_names}
            out = self.session.run(self._outputs, feed)
            logits_all.append(np.asarray(out[0], dtype=np.float32))
            if len(out) > 1:
                pooled_all.append(np.asarray(out[1], dtype=np.float32))
        logits = np.concatenate(logits_all)
        pooled = np.concatenate(pooled_all) if pooled_all else None
        if pooled is not None and self._emb_mean is not None and self._emb_mean.shape[0] == pooled.shape[1]:
            pooled = pooled - self._emb_mean
        return logits, pooled

    def classify(self, segments: list[SegmentInput]) -> list[Verdict]:
        if not segments:
            return []
        try:
            per_seg = [self.encode(s) for s in segments]
            flat = [e for encs in per_seg for e in encs]
            logits, pooled = self._run(flat)
        except Exception as exc:
            raise ClassifierUnavailable(f"inference failed: {exc}") from exc
        if logits.shape[-1] != len(self.meta.labels):  # a mismatched model must never be guessed at
            raise ClassifierUnavailable(
                f"model returns {logits.shape[-1]} scores but its metadata names {len(self.meta.labels)} labels"
            )
        probs = softmax(logits, self.meta.temperature)
        verdicts: list[Verdict] = []
        offset = 0
        for seg, encs in zip(segments, per_seg, strict=True):
            n = len(encs)
            p = probs[offset : offset + n]
            decision, label, score, pdict = decide(p, self.meta.labels, self.table, seg.purpose)
            chunk_emb = [pooled[offset + i] for i in range(n)] if pooled is not None else []
            emb = np.mean(np.stack(chunk_emb), axis=0) if chunk_emb else None
            verdicts.append(Verdict(decision, label, score, pdict, n, emb, chunk_emb))
            offset += n
        return verdicts


# --------------------------------------------------------------------------- the missing-model stand-in


class UnavailableClassifier:
    """Stands in when no model could be loaded: every call raises, so callers fail closed."""

    name = "unavailable"
    version = "-"
    embedding_cosine: float | None = None

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def classify(self, segments: list[SegmentInput]) -> list[Verdict]:
        if not segments:
            return []
        raise ClassifierUnavailable(self.reason)


def load_classifier(model_dir: Path | None, *, fpr_target: float, providers: list[str] | None = None) -> Classifier:
    if model_dir is None or not Path(model_dir).exists():
        return UnavailableClassifier(f"classifier weights not found ({model_dir}); run `mcp-shield pull-model`")
    try:
        return OnnxClassifier(Path(model_dir), fpr_target=fpr_target, providers=providers)
    except ClassifierUnavailable as exc:
        return UnavailableClassifier(str(exc))
