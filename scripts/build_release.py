"""Build the release artefacts into the Packages folder, without touching the source tree.

    python scripts/build_release.py [--out ../../Packages] [--verify]

* copies ``project/`` (minus caches, training outputs and VCS files) to a temp directory
  and builds there, so no ``build/`` or ``*.egg-info`` appears in the source tree;
* writes the sdist and the wheel. The wheel is pure Python (``py3-none-any``): the same
  file installs on Windows, macOS and Linux, and pip pulls the platform wheels of
  ONNX Runtime, tokenizers and numpy for each OS;
* writes ``SHA256SUMS`` next to them;
* ``--verify`` installs the wheel into a throw-away virtual environment that reuses the
  already-installed dependencies (no network), then runs ``mcp-shield --version`` and
  ``mcp-shield rules --print --short``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns(
    ".git",
    ".github",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    "*.egg-info",
    "data",
    "raw",
    "heldout",
    "runs",
    "models",
    "loso",
    "*.onnx",
)


def build(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mcp-shield-build-") as tmp:
        src = Path(tmp) / "project"
        shutil.copytree(PROJECT, src, ignore=IGNORE)
        dist = Path(tmp) / "dist"
        dist.mkdir()
        try:
            import build as _build  # noqa: F401

            subprocess.run(
                [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist), str(src)], check=True, cwd=tmp
            )
        except ImportError:  # the PEP 517 hooks of the declared backend, called directly
            hooks = f"import setuptools.build_meta as b; b.build_sdist({str(dist)!r}); b.build_wheel({str(dist)!r})"
            subprocess.run([sys.executable, "-c", hooks], check=True, cwd=src)
        made = []
        for f in sorted(dist.iterdir()):
            target = out / f.name
            shutil.copy2(f, target)
            made.append(target)
    return made


def checksums(files: list[Path], out: Path) -> Path:
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files]
    path = out / "SHA256SUMS"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="mcp-shield-verify-") as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
        windows = sys.platform == "win32"
        py = venv / ("Scripts" if windows else "bin") / ("python.exe" if windows else "python")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--no-deps", "--no-index", str(wheel)], check=True)
        exe = py.with_name("mcp-shield.exe" if sys.platform == "win32" else "mcp-shield")
        for args in (["--version"], ["rules", "--print", "--short"]):
            r = subprocess.run([str(exe), *args], capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise SystemExit(f"verification failed: mcp-shield {' '.join(args)}\n{r.stderr}")
            print(f"  mcp-shield {' '.join(args)} -> ok ({r.stdout.splitlines()[0][:60]})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=PROJECT.parent / "Packages")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    files = build(args.out)
    sums = checksums(files, args.out)
    for f in [*files, sums]:
        print(f"{f}  ({f.stat().st_size / 1024:.0f} KB)")
    if args.verify:
        wheel = next(f for f in files if f.suffix == ".whl")
        verify(wheel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
