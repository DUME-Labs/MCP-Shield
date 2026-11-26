"""Opt-in: does the train → export → install path work end to end? (pytest --run-training-code)

Uses the toy fixture and a tiny randomly initialised BERT (the hidden ``--smoke`` switch),
entirely inside the test's temp folder; nothing is kept. It checks the code, not a model.
"""

from __future__ import annotations

import pytest

from helpers import toy_dataset

pytestmark = [pytest.mark.training_code, pytest.mark.slow]


def test_train_export_install(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mcp_shield import models
    from mcp_shield.cascade.tier3_classifier import OnnxClassifier, SegmentInput
    from mcp_shield.config import SecurityConfig
    from mcp_shield_train.__main__ import main
    from mcp_shield_train.data import write_jsonl
    from mcp_shield_train.split import grouped_split

    records = toy_dataset.generate(360)
    grouped_split(records, seed=1)
    data = tmp_path / "data"
    for split in ("train", "val", "test_internal"):
        write_jsonl(data / f"{split}.jsonl", [r for r in records if r.split == split])
    run, out = tmp_path / "run", tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--data",
                str(data),
                "--out",
                str(run),
                "--smoke",
                "--epochs",
                "2",
                "--max-epochs",
                "2",
                "--lr",
                "1e-3",
                "--max-length",
                "64",
                "--stride",
                "8",
                "--batch-size",
                "16",
                "--grad-accum",
                "1",
            ]
        )
        == 0
    )
    assert (run / "checkpoint").exists()
    assert main(["export", "--run", str(run), "--data", str(data), "--out", str(out), "--version", "toy"]) == 0
    assert models.verify_dir(out) == []
    clf = OnnxClassifier(out)
    v = clf.classify([SegmentInput("[server=web tool=fetch field=content_0_.text]", "hello world")])[0]
    assert v.decision in ("allow", "abstain", "block") and v.embedding is not None
    target = models.install_from(out, SecurityConfig())
    assert models.verify_dir(target) == []
    assert main(["evaluate", "--model", str(out), "--data", str(data / "test_internal.jsonl")]) == 0
