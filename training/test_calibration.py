"""Temperature scaling, FPR thresholds, length-adjusted tables, embedding calibration."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_shield.cascade.tier3_classifier import ThresholdTable
from mcp_shield_train.calibrate import ValDoc, calibrate, embedding_calibration, fit_temperature
from mcp_shield_train.metrics import attack_prob, expected_calibration_error, softmax, threshold_at_fpr


def test_threshold_at_fpr_property():
    rng = np.random.default_rng(0)
    for n in (10, 199, 1000, 4321):
        scores = rng.random(n)
        for fpr in (0.0, 0.005, 0.02, 0.1):
            t = threshold_at_fpr(scores, fpr)
            assert (scores >= t).mean() <= fpr + 1e-12
            assert (scores >= np.nextafter(t, -np.inf) - 1e-9).mean() >= (scores >= t).mean()


def test_temperature_recovered():
    rng = np.random.default_rng(1)
    n, true_t = 4000, 2.5
    labels = rng.integers(0, 4, size=n)
    base = rng.normal(size=(n, 4))
    base[np.arange(n), labels] += 3.0
    sampled = np.array([rng.choice(4, p=p) for p in softmax(base, 1.0)])
    t = fit_temperature(base * true_t, sampled)
    assert t == pytest.approx(true_t, rel=0.1)


def test_temperature_improves_calibration():
    rng = np.random.default_rng(2)
    n = 3000
    labels = rng.integers(0, 4, size=n)
    logits = rng.normal(size=(n, 4))
    logits[np.arange(n), labels] += 2.0
    over = logits * 4.0  # an over-confident model
    t = fit_temperature(over, labels)
    p_raw, p_cal = softmax(over, 1.0), softmax(over, t)
    correct = p_raw.argmax(1) == labels
    assert expected_calibration_error(p_cal.max(1), correct) < expected_calibration_error(p_raw.max(1), correct)


def make_docs(rng, n_benign=1500, n_attack=300, long_share=0.2):
    docs = []
    for i in range(n_benign):
        chunks = rng.integers(2, 12) if rng.random() < long_share else 1
        lg = rng.normal(size=(chunks, 4))
        lg[:, 0] += 4.0
        docs.append(ValDoc(lg, "benign", hard_negative=i % 2 == 0, group_id=f"b{i}"))
    for i in range(n_attack):
        lg = rng.normal(size=(1, 4))
        lg[:, 1] += 5.0
        docs.append(ValDoc(lg, "indirect_injection", group_id=f"a{i}"))
    for i in range(60):
        lg = rng.normal(size=(1, 4))
        lg[:, 0] += 3.0
        docs.append(ValDoc(lg, "benign", purpose="description", group_id=f"d{i}"))
    return docs


def test_calibrate_produces_runtime_tables():
    rng = np.random.default_rng(3)
    cal = calibrate(make_docs(rng))
    meta = cal.to_meta()
    assert set(meta["thresholds"]) == {"0.005", "0.01", "0.02"}
    for _key, entry in meta["thresholds"].items():
        table = ThresholdTable.from_json(entry)  # the engine can load it
        blocks = [table.block_threshold(n) for n in (1, 2, 4, 8, 16, 32)]
        assert blocks == sorted(blocks), "longer documents never get a looser threshold"
        assert all(table.abstain_threshold(n) <= table.block_threshold(n) for n in (1, 4, 32))
        assert table.block_threshold(1, purpose="description") <= table.block_threshold(1)
    strict = ThresholdTable.from_json(meta["thresholds"]["0.005"]).block_threshold(1)
    loose = ThresholdTable.from_json(meta["thresholds"]["0.02"]).block_threshold(1)
    assert strict >= loose
    assert cal.report["recall@0.005"]["indirect_injection"] > 0.5


def test_per_document_fpr_holds_on_long_documents():
    rng = np.random.default_rng(4)
    docs = make_docs(rng, n_benign=4000, long_share=0.5)
    cal = calibrate(docs, fpr_targets=(0.02,))
    table = ThresholdTable.from_json(cal.thresholds["0.02"])
    long_docs = [d for d in docs if d.label == "benign" and d.purpose == "content" and len(d.chunk_logits) >= 4]
    flagged = [
        float(np.max(attack_prob(softmax(d.chunk_logits, cal.temperature))))
        >= table.block_threshold(len(d.chunk_logits))
        for d in long_docs
    ]
    assert np.mean(flagged) <= 0.03


def test_embedding_calibration():
    rng = np.random.default_rng(5)
    spread = rng.normal(size=(400, 16))
    groups = [f"g{i}" for i in range(400)]
    cal = embedding_calibration(spread, groups)
    assert cal is not None and 0.9 <= cal["cosine"] <= 0.99 and len(cal["mean"]) == 16
    # two tight clusters: even after centring, unrelated texts in one cluster look identical → rule off
    centres = rng.normal(size=(2, 16)) * 5
    clustered = centres[rng.integers(0, 2, size=400)] + rng.normal(scale=1e-3, size=(400, 16))
    assert embedding_calibration(clustered, groups) is None
    assert embedding_calibration(spread[:10], groups[:10]) is None, "too few vectors to calibrate"
