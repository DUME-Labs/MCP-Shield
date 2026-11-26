"""Classifier weights: where they live, how they are fetched and verified (07 §1, §6).

Weights are hosted on the Hugging Face Hub and downloaded once into the ``platformdirs``
cache. Every file is checked against a SHA-256 shipped **in this package** (a checksum
fetched from the same place as the weights would prove nothing). Until a release is
pinned here, only a locally built model can be installed (``pull-model --from DIR``) or
an unpinned download explicitly allowed for development (``--allow-unpinned``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .cascade.tier3_classifier import META_FILE, Classifier, OnnxClassifier, UnavailableClassifier
from .config import SecurityConfig
from .paths import data_dir, models_dir, write_private_file

RUNTIME_FILE = "runtime.json"
INSTALLED_FILE = "installed.json"
MODEL_FILES = ("model.onnx", "model.fp16.onnx", "tokenizer.json", META_FILE)
REQUIRED_FILES = ("model.onnx", "tokenizer.json", META_FILE)

# Pinned releases. Filled in when a model version is published; ``None`` means "not yet".
REGISTRY: dict[str, dict[str, Any]] = {
    "mcp-shield-clf": {
        "repo": "rishikesh/mcp-shield-clf",
        "revision": None,
        "sha256": {},
    },
}


class ModelError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(config: SecurityConfig) -> Path:
    if config.model_dir:
        return Path(config.model_dir).expanduser()
    entry = REGISTRY.get(config.classifier_model, {})
    return models_dir() / config.classifier_model / str(entry.get("revision") or "local")


def verify_dir(directory: Path) -> list[str]:
    """Problems with an installed model directory (empty when it is intact)."""
    problems: list[str] = []
    for name in REQUIRED_FILES:
        if not (directory / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems
    expected: dict[str, str] = {}
    installed = directory / INSTALLED_FILE
    if installed.exists():
        expected.update(json.loads(installed.read_text(encoding="utf-8")).get("sha256", {}))
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    for name, digest in dict(meta.get("files", {})).items():
        expected.setdefault(name, digest)
    for name, digest in expected.items():
        path = directory / name
        if not path.exists():
            problems.append(f"missing {name}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum mismatch for {name}")
    return problems


def install_from(source: Path, config: SecurityConfig) -> Path:
    """Install a locally built model directory (e.g. ``training/export_onnx.py`` output)."""
    source = Path(source)
    missing = [n for n in REQUIRED_FILES if not (source / n).exists()]
    if missing:
        raise ModelError(f"{source} is missing {', '.join(missing)}")
    target = model_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in MODEL_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
            digests[name] = sha256_file(target / name)
    write_private_file(
        target / INSTALLED_FILE, json.dumps({"source": str(source), "sha256": digests}, indent=2).encode()
    )
    problems = verify_dir(target)
    if problems:
        raise ModelError("installed model failed verification: " + "; ".join(problems))
    return target


def pull(
    config: SecurityConfig, *, allow_unpinned: bool = False, token: str | None = None, include_fp16: bool = False
) -> Path:
    """Download the pinned release from the Hub and verify every file."""
    entry = REGISTRY.get(config.classifier_model)
    if entry is None:
        raise ModelError(f"unknown model {config.classifier_model!r}")
    revision, pinned = entry.get("revision"), dict(entry.get("sha256") or {})
    if not revision and not allow_unpinned:
        raise ModelError(
            f"no release of {config.classifier_model} is pinned in this version of MCP-Shield yet; install a locally "
            "built model with `mcp-shield pull-model --from <dir>`, or pass --allow-unpinned for development"
        )
    from huggingface_hub import hf_hub_download

    target = model_dir(config)
    target.mkdir(parents=True, exist_ok=True)
    names = [n for n in MODEL_FILES if include_fp16 or n != "model.fp16.onnx"]
    digests = {}
    for name in names:
        try:
            path = Path(
                hf_hub_download(entry["repo"], name, revision=revision or "main", local_dir=str(target), token=token)
            )
        except Exception as exc:
            if name == "model.fp16.onnx":
                continue
            raise ModelError(f"download of {name} failed: {exc}") from exc
        digest = sha256_file(path)
        if pinned.get(name) and pinned[name] != digest:
            path.unlink(missing_ok=True)
            raise ModelError(f"{name}: checksum {digest} does not match the pinned {pinned[name]} — file removed")
        digests[name] = digest
    write_private_file(
        target / INSTALLED_FILE,
        json.dumps(
            {"source": f"hf://{entry['repo']}@{revision or 'main'}", "pinned": bool(revision), "sha256": digests},
            indent=2,
        ).encode(),
    )
    return target


# --------------------------------------------------------------------------- execution providers


def saved_runtime() -> dict[str, Any]:
    try:
        return dict(json.loads((data_dir() / RUNTIME_FILE).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def save_runtime(data: dict[str, Any]) -> None:
    write_private_file(data_dir() / RUNTIME_FILE, json.dumps(data, indent=2).encode())


def preferred_providers() -> list[str]:
    """The execution provider ``doctor``'s self-test chose, if still available; else CPU."""
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return ["CPUExecutionProvider"]
    chosen = [p for p in saved_runtime().get("providers", []) if p in available]
    if chosen and chosen[-1] != "CPUExecutionProvider":
        chosen.append("CPUExecutionProvider")
    return chosen or ["CPUExecutionProvider"]


def load_configured_classifier(config: SecurityConfig) -> Classifier:
    directory = model_dir(config)
    if not directory.exists():
        return UnavailableClassifier(f"classifier weights not found in {directory}; run `mcp-shield pull-model`")
    problems = verify_dir(directory)
    if problems:
        return UnavailableClassifier(f"model in {directory} failed verification: {'; '.join(problems)}")
    try:
        return OnnxClassifier(directory, fpr_target=config.fpr_target, providers=preferred_providers())
    except Exception as exc:
        return UnavailableClassifier(str(exc))
