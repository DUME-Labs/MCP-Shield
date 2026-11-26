"""A tiny but real ONNX classifier + tokenizer with the production interface.

The graph has the shipped signature — inputs ``input_ids`` / ``attention_mask``, outputs
``logits [B, 4]`` and ``pooled [B, 8]`` — and a WordLevel ``tokenizer.json`` with a
``[CLS] A [SEP] B [SEP]`` template, so tokenisation, pair truncation with overflow,
batching, temperature, the threshold table and fail-closed loading are all exercised
without the 300 MB model. Its "knowledge" is a bag of words: each attack word pushes
the logits towards one attack class.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
INDIRECT = [
    "ignore",
    "previous",
    "instructions",
    "assistant",
    "system",
    "override",
    "exfiltrate",
    "id_rsa",
    "secretly",
    "disregard",
    "forward",
    "credentials",
]
COMMAND = ["rm", "rf", "curl", "wget", "bash", "chmod", "nc", "sudo", "mkfifo"]
POISON = ["hidden", "always", "before", "silently", "conceal"]
MILD = ["perhaps", "maybe", "kindly"]  # lands in the abstain band
BENIGN = [
    "the",
    "a",
    "build",
    "release",
    "notes",
    "file",
    "report",
    "meeting",
    "issue",
    "server",
    "data",
    "hello",
    "world",
    "summary",
    "project",
    "weather",
    "today",
    "is",
    "sunny",
    "and",
    "of",
    "to",
    "in",
]
LABELS = ["benign", "indirect_injection", "command_injection", "description_poisoning"]


def vocab() -> dict[str, int]:
    words = SPECIALS + INDIRECT + COMMAND + POISON + MILD + BENIGN
    return {w: i for i, w in enumerate(dict.fromkeys(words))}


def _tokenizer_json(v: dict[str, int]) -> dict:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": v[t],
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for t in SPECIALS
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ],
            "special_tokens": {t: {"id": t, "ids": [v[t]], "tokens": [t]} for t in ("[CLS]", "[SEP]")},
        },
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": v, "unk_token": "[UNK]"},
    }


def _embedding(v: dict[str, int]) -> np.ndarray:
    e = np.zeros((len(v), 4), dtype=np.float32)
    for w in INDIRECT:
        e[v[w], 1] = 4.0
    for w in COMMAND:
        e[v[w], 2] = 4.0
    for w in POISON:
        e[v[w], 3] = 4.0
    for w in MILD:
        e[v[w], 1] = 2.0
    return e


def build(
    directory: Path,
    *,
    max_length: int = 32,
    stride: int = 4,
    block: float = 0.9,
    abstain: float = 0.6,
    with_checksums: bool = True,
) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory.mkdir(parents=True, exist_ok=True)
    v = vocab()
    emb = _embedding(v)
    proj = np.random.default_rng(7).normal(size=(4, 8)).astype(np.float32)
    bias = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    nodes = [
        helper.make_node("Gather", ["E", "input_ids"], ["tok"], axis=0),
        helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["maskf", "axis2"], ["mask3"]),
        helper.make_node("Mul", ["tok", "mask3"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
        helper.make_node("Add", ["summed", "bias"], ["logits"]),
        helper.make_node("MatMul", ["summed", "P"], ["pooled_raw"]),
        helper.make_node("Add", ["pooled_raw", "pbias"], ["pooled"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_mcp_shield_clf",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 4]),
            helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch", 8]),
        ],
        initializer=[
            numpy_helper.from_array(emb, "E"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(proj, "P"),
            numpy_helper.from_array(np.full(8, 0.5, np.float32), "pbias"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axis2"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="mcp-shield-tests")
    model.ir_version = 8
    onnx.checker.check_model(model)
    (directory / "model.onnx").write_bytes(model.SerializeToString())
    (directory / "tokenizer.json").write_text(json.dumps(_tokenizer_json(v)), encoding="utf-8")
    meta = {
        "name": "mcp-shield-clf",
        "version": "tiny-test",
        "labels": LABELS,
        "temperature": 1.0,
        "max_length": max_length,
        "stride": stride,
        "thresholds": {
            "0.005": {
                "block": {"1": block, "4": min(0.99, block + 0.04)},
                "abstain": {"1": abstain},
                "class_block": {"description_poisoning": {"1": block - 0.1}},
            },
            "0.02": {"block": {"1": block - 0.2}, "abstain": {"1": abstain - 0.2}},
        },
    }
    if with_checksums:
        meta["files"] = {
            n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in ("model.onnx", "tokenizer.json")
        }
    (directory / "mcp_shield_clf.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return directory
