"""Fine-tuning ``mcp-shield-clf`` (03 §7).

Full fine-tune of ``microsoft/mdeberta-v3-base`` with class-weighted cross-entropy
(inverse frequency, capped) and label smoothing 0.05; AdamW, lr 2e-5, 6 % warm-up then
linear decay, weight decay 0.01, fp16 on a GPU; early stopping on **recall at FPR 0.5 %**
(document-level, on validation) — the operating point actually deployed. Inputs are the
pair (context prefix, text) chunked exactly as the runtime chunks them.

``--smoke`` swaps the backbone for a tiny randomly initialised BERT and a WordLevel
tokenizer built from the data, so the whole pipeline runs offline on a CPU in minutes.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import LABEL_ID, LABELS
from .data import Record
from .metrics import attack_prob, recall_at_fpr, softmax


@dataclass
class TrainConfig:
    base_model: str = "microsoft/mdeberta-v3-base"
    max_length: int = 512
    stride: int = 64
    lr: float = 2e-5
    epochs: int = 3
    max_epochs: int = 5
    batch_size: int = 16
    grad_accum: int = 2
    warmup: float = 0.06
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    class_weight_cap: float = 10.0
    early_stop_fpr: float = 0.005
    patience: int = 1
    seed: int = 42
    fp16: bool = True
    smoke: bool = False


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smoke_tokenizer(records: list[Record], max_length: int) -> Any:
    """A WordLevel tokenizer over the training vocabulary with a [CLS] A [SEP] B [SEP] template."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"], min_frequency=1)
    tok.train_from_iterator((s for r in records for s in (r.context, r.text)), trainer)
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", cls), ("[SEP]", sep)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=max_length,
    )


def load_backbone(cfg: TrainConfig, train_records: list[Record]) -> tuple[Any, Any]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
        BertForSequenceClassification,
    )

    id2label = dict(enumerate(LABELS))
    if cfg.smoke:
        tokenizer = smoke_tokenizer(train_records, cfg.max_length)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=max(cfg.max_length, 128),
            num_labels=4,
            id2label=id2label,
            label2id=LABEL_ID,
        )
        return tokenizer, BertForSequenceClassification(config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=len(LABELS), id2label=id2label, label2id=LABEL_ID
    )
    return tokenizer, model


def encode(tokenizer: Any, records: list[Record], cfg: TrainConfig) -> tuple[list[dict[str, list[int]]], list[int]]:
    """Chunks (with the prefix on every chunk) and, per chunk, the index of its record."""
    enc = tokenizer(
        [r.context for r in records],
        [r.text for r in records],
        truncation="only_second",
        max_length=cfg.max_length,
        stride=cfg.stride,
        return_overflowing_tokens=True,
    )
    owners = list(enc["overflow_to_sample_mapping"])
    chunks = [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(owners))
    ]
    return chunks, owners


def class_weights(records: list[Record], cap: float) -> list[float]:
    counts = np.array([sum(1 for r in records if r.label == lab) for lab in LABELS], dtype=np.float64)
    weights = counts.sum() / np.maximum(counts, 1) / len(LABELS)
    weights = np.minimum(weights, cap)
    return list(weights / weights.mean())


def _batches(
    chunks: list[dict[str, list[int]]], labels: list[int], size: int, pad_id: int, shuffle: bool, rng: random.Random
) -> Any:
    import torch

    order = list(range(len(chunks)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        width = max(len(chunks[i]["input_ids"]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            n = len(chunks[i]["input_ids"])
            ids[row, :n] = torch.tensor(chunks[i]["input_ids"])
            mask[row, :n] = torch.tensor(chunks[i]["attention_mask"])
        yield idx, ids, mask, torch.tensor([labels[i] for i in idx], dtype=torch.long)


def predict(model: Any, tokenizer: Any, records: list[Record], cfg: TrainConfig, device: str) -> list[np.ndarray]:
    """Per-record chunk logits."""
    import torch

    chunks, owners = encode(tokenizer, records, cfg)
    out: list[list[np.ndarray]] = [[] for _ in records]
    model.eval()
    with torch.no_grad():
        for idx, ids, mask, _ in _batches(
            chunks, [0] * len(chunks), 64, tokenizer.pad_token_id, False, random.Random()
        ):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu().numpy()
            for row, i in enumerate(idx):
                out[owners[i]].append(logits[row])
    return [np.stack(x) for x in out]


def doc_recall_at_fpr(chunk_logits: list[np.ndarray], records: list[Record], fpr: float) -> tuple[float, float]:
    scores = np.array([float(np.max(attack_prob(softmax(c)))) for c in chunk_logits])
    benign = np.array([r.label == "benign" for r in records])
    if benign.all() or not benign.any():
        return 0.0, 1.0
    return recall_at_fpr(scores[benign], scores[~benign], fpr)


def train(train_records: list[Record], val_records: list[Record], out_dir: Path, cfg: TrainConfig) -> dict[str, Any]:
    import torch

    if not 0 <= cfg.stride <= cfg.max_length // 4:
        raise ValueError(
            f"stride {cfg.stride} must be between 0 and max_length/4 ({cfg.max_length // 4}): "
            "every chunk also carries the context prefix and special tokens"
        )
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_backbone(cfg, train_records)
    model.to(device)
    chunks, owners = encode(tokenizer, train_records, cfg)
    labels = [LABEL_ID[train_records[o].label] for o in owners]
    weights = torch.tensor(class_weights(train_records, cfg.class_weight_cap), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(chunks) / (cfg.batch_size * cfg.grad_accum))
    total = steps_per_epoch * cfg.max_epochs
    warm = max(1, int(cfg.warmup * steps_per_epoch * cfg.epochs))

    def lr_lambda(step: int) -> float:  # linear warm-up, then linear decay to zero
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total - step) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    use_amp = cfg.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(cfg.seed)
    best, best_epoch, history, bad = -1.0, -1, [], 0
    ckpt = out_dir / "checkpoint"
    for epoch in range(cfg.max_epochs):
        model.train()
        t0, running, step = time.time(), 0.0, 0
        for n, (_, ids, mask, y) in enumerate(
            _batches(chunks, labels, cfg.batch_size, tokenizer.pad_token_id, True, rng)
        ):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = loss_fn(logits.float(), y.to(device)) / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            if (n + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1
        val_logits = predict(model, tokenizer, val_records, cfg, device)
        recall, threshold = doc_recall_at_fpr(val_logits, val_records, cfg.early_stop_fpr)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": running / max(1, n + 1),
                "val_recall_at_fpr": recall,
                "threshold": threshold,
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(json.dumps(history[-1]))
        if recall > best:
            best, best_epoch, bad = recall, epoch + 1, 0
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
        else:
            bad += 1
        if epoch + 1 >= cfg.epochs and bad >= cfg.patience:  # past the minimum epochs: continue only while improving
            break
    summary = {
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_recall_at_fpr": best,
        "history": history,
        "train_records": len(train_records),
        "train_chunks": len(chunks),
        "device": device,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
