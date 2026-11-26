"""Training pipeline for ``mcp-shield-clf`` (vault doc 03).

Order matters (03 §5): global deduplication *within* each label → grouping → grouped,
stratified split → augmentation on train only → training → temperature scaling and
threshold selection on validation → ONNX export (logits + pooled outputs, int8) with the
metadata the runtime reads. Held-out benchmarks never enter any of these steps.

Run ``python -m mcp_shield_train --help`` from the ``training`` directory.
"""

LABELS = ("benign", "indirect_injection", "command_injection", "description_poisoning")
LABEL_ID = {name: i for i, name in enumerate(LABELS)}
__all__ = ["LABELS", "LABEL_ID"]
