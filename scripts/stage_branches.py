"""Stage the contents of each GitHub branch into ``release-branches/<branch>`` (nothing is pushed).

    python scripts/stage_branches.py

* ``main``    — the project at the root (pyproject, src, training, scripts), README, LICENSE,
                NOTICE, CONTRIBUTING, SECURITY, ``docs/`` (design docs, host prompt rules,
                figures, the IEEE paper) and the CI + release workflows;
* ``tests``   — the test suite at the root, with its own workflow that checks ``main`` out
                beside it;
* ``dataset`` — the contribution guide, an empty ``contributions/`` folder and the PR check.

Private material (``docs/extras``: reports, templates, the base-paper PDF, review notes)
is never copied. Obsidian ``[[links]]`` become ordinary Markdown links, and the base paper
is cited by its arXiv link. Re-running regenerates the three folders from the sources.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]  # F:\MCP-Shield
PROJECT, TESTS, DOCS = ROOT / "project", ROOT / "Tests", ROOT / "docs"
OUT = ROOT / "release-branches"
BASE_PAPER = "https://arxiv.org/abs/2601.17549"
REPO = "https://github.com/DUME-Labs/MCP-Shield"
SKIP = shutil.ignore_patterns(
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
PUBLIC_DOCS = [
    "00 - Master Architecture & Overview.md",
    "01 - Threat Model & Vulnerabilities.md",
    "02 - System Architecture & Technical Implementation.md",
    "03 - Fine-Tuning & Dataset Strategy.md",
    "04 - Brainstorming Archive & Trade-off Analysis.md",
    "05 - Literature Survey & Research Papers.md",
    "06 - Fortified Solution & Defense Upgrades.md",
    "07 - Model Distribution, PyPI Packaging & Local Dashboard.md",
    "08 - Build Roadmap - Stage 2 & Enterprise.md",
    "Host Prompt Rules.md",
    "MCP-Shield Documentation.md",
    "README.md",
]
PAPER = DOCS / "extras" / "MCP-Shield IEEE Paper.pdf"


# --------------------------------------------------------------------------- docs conversion


def link(target: str) -> str:
    return quote(target, safe="/#")


def convert_markdown(text: str, available: set[str]) -> str:
    """Obsidian wiki links → Markdown links; the local base-paper copy → its arXiv link."""
    text = text.replace("Local copy: [[Breaking the Protocol.pdf]]", f"Available at {BASE_PAPER}")
    text = re.sub(r"!?\[\[Breaking the Protocol\.pdf(?:\|[^\]]*)?\]\]", f"[the base paper]({BASE_PAPER})", text)

    def wiki(m: re.Match[str]) -> str:
        embed, target, alias = m.group(1), m.group(2).strip(), m.group(3)
        name, _, anchor = target.partition("#")
        label = (alias or name).strip()
        if name.lower().endswith((".png", ".jpg", ".svg")):
            return f"![{label}](images/{link(name)})"
        file = name if name.endswith(".md") else f"{name}.md"
        if file in available:
            return f"[{label}]({link(file)}{'#' + link(anchor.lower().replace(' ', '-')) if anchor else ''})"
        _ = embed
        return label  # a note that is not published: keep the words, drop the link

    text = re.sub(r"(!?)\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", wiki, text)
    for name, rel in (
        ("LICENSE.md", "../LICENSE"),
        ("NOTICE.md", "../NOTICE"),
        ("CONTRIBUTING.md", "../CONTRIBUTING.md"),
        ("SECURITY.md", "../SECURITY.md"),
    ):
        text = text.replace(f"]({name})", f"]({rel})")
    return text


# --------------------------------------------------------------------------- generated files

CI_STEPS = """      - uses: actions/setup-python@v5
        with:
          python-version: ${{{{ matrix.python }}}}
      - name: Install
        run: python -m pip install -e "./project[dev]"
      - name: Tests
        working-directory: Tests
        run: python -m pytest -ra
      - name: Lint and types
        if: matrix.python == '3.13'
        working-directory: project
        run: |
          python -m ruff check src training scripts
          python -m mypy src
"""

CI_MATRIX = """    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python: ["3.10", "3.13"]
    runs-on: ${{ matrix.os }}
"""

MAIN_CI = f"""name: ci
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
permissions:
  contents: read
jobs:
  test:
{CI_MATRIX}    steps:
      - uses: actions/checkout@v4
        with:
          path: project
      - uses: actions/checkout@v4
        with:
          ref: tests
          path: Tests
{CI_STEPS.format()}"""

TESTS_CI = f"""name: tests
on:
  push:
    branches: [tests]
  pull_request:
    branches: [tests]
permissions:
  contents: read
jobs:
  test:
{CI_MATRIX}    steps:
      - uses: actions/checkout@v4
        with:
          path: Tests
      - uses: actions/checkout@v4
        with:
          ref: main
          path: project
{CI_STEPS.format()}"""

RELEASE = """name: release
on:
  push:
    tags: ["v*"]
permissions:
  contents: write
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build the wheel and sdist (one universal wheel for Windows, macOS and Linux)
        run: |
          python -m pip install build .
          python scripts/build_release.py --out dist --verify
      - name: Attach them to a GitHub release
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh release create "$GITHUB_REF_NAME" dist/* --title "MCP-Shield $GITHUB_REF_NAME" --generate-notes --prerelease
"""

DATASET_CI = """name: check-contributions
on:
  pull_request:
    branches: [dataset]
  push:
    branches: [dataset]
permissions:
  contents: read
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          path: dataset
      - uses: actions/checkout@v4
        with:
          ref: main
          path: tooling
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install ./tooling
      - name: Check every contribution file
        working-directory: tooling/training
        run: python -m mcp_shield_train contributions check "$GITHUB_WORKSPACE/dataset/contributions"
"""

CODEOWNERS = r"""# Changes to these paths need a code owner's review once branch protection requires it.
# Add teammates' GitHub handles after the existing one, separated by spaces.

/docs/Host\ Prompt\ Rules.md        @rishikeshvarma
/src/mcp_shield/host_rules.md       @rishikeshvarma
/src/mcp_shield/cascade/            @rishikeshvarma
/src/mcp_shield/engine/             @rishikeshvarma
/src/mcp_shield/proxy/              @rishikeshvarma
/src/mcp_shield/taint.py            @rishikeshvarma
/src/mcp_shield/pinning.py          @rishikeshvarma
/src/mcp_shield/dashboard/          @rishikeshvarma
/.github/                           @rishikeshvarma
"""

CONTRIBUTING = f"""# Contributing to MCP-Shield

Thank you for helping. MCP-Shield is a student-led research project and security software,
so a few rules matter more than usual.

## Ground rules

- **Numbers are measured or labelled.** Latency, ASR, FPR and footprint figures are targets
  until the benchmark phase measures them. Do not add an unmeasured number as a result.
- **Security-sensitive paths get careful review:** `src/mcp_shield/cascade/`, `engine/`,
  `proxy/`, `taint.py`, `pinning.py`, `store.py`, `spotlight.py`, `dashboard/` and the host
  prompt rules. A subtle bug there is a bypass.
- **No secrets, ever** — not in code, tests, fixtures or issues.
- **Report vulnerabilities privately** ([SECURITY.md](SECURITY.md)), never in a public issue.

## Branches

| Branch | For |
|:--|:--|
| `main` | code, docs and the paper — open pull requests here |
| `tests` | the test suite — a change to behaviour comes with a pull request here too |
| `dataset` | classifier training data — see [training/CONTRIBUTING-DATA.md](training/CONTRIBUTING-DATA.md) |

Training examples (including attack examples) belong on the `dataset` branch only, in the
format that guide describes. The team's adaptive red-team set is not published.

## Development setup

```bash
git clone {REPO}.git && cd MCP-Shield
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
git worktree add ../MCP-Shield-tests tests
cd ../MCP-Shield-tests && MCP_SHIELD_PROJECT=../MCP-Shield python -m pytest
```

Python 3.10–3.13 on Windows, macOS or Linux. CI runs the suite on all three.

## Pull requests

1. One logical change per pull request; a design change and its implementation are two.
2. CI must be green: `pytest` (from the `tests` branch), `ruff check src training scripts`,
   `mypy src`.
3. Detection changes need a positive case (the attack is caught) **and** a benign twin (the
   same shape without the attack is allowed) — false positives are bugs too.
4. **Host prompt rules** change only in `docs/Host Prompt Rules.md`, with a new changelog row
   and the same file copied to `src/mcp_shield/host_rules.md`; a test fails if they differ.
   Never renumber rules — add new ones at the end.

## Licence

By contributing code you agree that it is released under the Apache-2.0 licence of this
repository.
"""

TESTS_README = """# MCP-Shield — test suite

This branch holds the tests; the code is on `main`. Check `main` out next to it:

```bash
git clone https://github.com/DUME-Labs/MCP-Shield.git MCP-Shield
git -C MCP-Shield worktree add ../Tests tests
pip install -e "./MCP-Shield[dev]"
cd Tests && MCP_SHIELD_PROJECT=../MCP-Shield python -m pytest
```

Without `MCP_SHIELD_PROJECT` the suite looks for the code in `../project`.

| Folder | What it covers |
|:--|:--|
| `unit/` | each module: normalisation, protocol, config, winnowing, argument policy, Tiers 1–3, extractor, Spotlighting, redaction, store, taint, pinning, install, models, CLI, host-rule integrity |
| `integration/` | the engine's decisions message by message, IPC, the HTTP listener, the dashboard's security |
| `e2e/` | real processes: a front-end, an autostarted engine and a fake stdio MCP server |
| `training/` | the dataset pipeline and contributions; `--run-training-code` also runs train → export on a toy model |
| `helpers/`, `fakes/` | a tiny real ONNX classifier, a toy record set, a fake MCP server |

`pytest -m "not slow"` skips the tests that start processes.
"""

CONTRIBUTIONS_README = """# contributions

One file per push: `<name>-<NNN>.jsonl` (for example `rishikesh-001.jsonl`).
The format and the rules are in the branch's README. Check a file before pushing:

```bash
python -m mcp_shield_train contributions check contributions/<your file>
```
"""


GIT_COMMANDS = f"""# Publishing the staged branches

Everything under `release-branches/` is ready to commit. Nothing has been pushed.
The commands below are for **Git Bash**. In the Windows Command Prompt (cmd) or PowerShell,
`cp -r /f/MCP-Shield/release-branches/<branch>/. .` becomes
`robocopy F:\\MCP-Shield\\release-branches\\<branch> . /E` (robocopy's exit code 1 means
"files copied", not an error); every git command is the same.

## 0. Clone the repository

```bash
git clone {REPO}.git MCP-Shield-repo
cd MCP-Shield-repo
```

## 1. Keep today's `main` as `legacy-main`

```bash
git branch legacy-main origin/main
git push origin legacy-main
```

## 2. Replace `main` with the project

```bash
git checkout main
git rm -r -q .
cp -r /f/MCP-Shield/release-branches/main/. .
git add -A
git status            # review: no extras, no reports, no data
git commit -m "MCP-Shield Stage 1: proxy, engine, cascade, dashboard, training pipeline and docs"
git push origin main
```

## 3. Create `tests`

```bash
git checkout --orphan tests
git rm -r -q --cached . && git clean -fdq
cp -r /f/MCP-Shield/release-branches/tests/. .
git add -A
git commit -m "Test suite for MCP-Shield Stage 1"
git push -u origin tests
```

## 4. Create `dataset`

```bash
git checkout --orphan dataset
git rm -r -q --cached . && git clean -fdq
cp -r /f/MCP-Shield/release-branches/dataset/. .
git add -A
git commit -m "Dataset branch: contribution guide and checks"
git push -u origin dataset
git checkout main
```

## 5. On GitHub (Settings)

- **Branch protection** on `main`, `tests` and `dataset`: require pull requests and the CI
  checks (`ci`, `tests`, `check-contributions`) before merging.
- **CODEOWNERS:** fill in `.github/CODEOWNERS` with the maintainers' handles, then turn on
  "Require review from Code Owners" for `main`. This is what keeps the host prompt rules
  from changing without review.
- **Security:** enable private vulnerability reporting, secret scanning and push protection.
- Fill in the contact line of `SECURITY.md` if you want an e-mail fallback.

## 6. Releases ("Packages" on the side)

The wheel and sdist are built by the `release` workflow when you push a version tag:

```bash
git tag v0.1.0.dev0
git push origin v0.1.0.dev0
```

They are the same files as in `F:\\MCP-Shield\\Packages`, attached to the GitHub release.
"""


def fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


NAV = "**[Overview](OVERVIEW.md)** · [Contributing](CONTRIBUTING.md) · [License](LICENSE) · [Security](SECURITY.md)"


def front_page(available: set[str]) -> str:
    """The design overview (docs/README.md) as the repository's front page, with links fixed for the root."""
    text = convert_markdown((DOCS / "README.md").read_text(encoding="utf-8"), available)
    for rel, root in (
        ("../LICENSE", "LICENSE"),
        ("../NOTICE", "NOTICE"),
        ("../CONTRIBUTING.md", "CONTRIBUTING.md"),
        ("../SECURITY.md", "SECURITY.md"),
    ):
        text = text.replace(f"]({rel})", f"]({root})")
    for name in available - {"README.md"}:
        text = text.replace(f"]({link(name)})", f"](docs/{link(name)})")
    text = re.sub(r"\]\(images/", "](docs/images/", text)
    marker = "[Features](#-key-features)"
    if marker in text:
        return text.replace(marker, f"{NAV}\n\n{marker}", 1)
    return f"{NAV}\n\n{text}"


def stage_main() -> Path:
    main = fresh(OUT / "main")
    for name in ("pyproject.toml", "LICENSE", "NOTICE", ".gitignore"):
        shutil.copy2(PROJECT / name, main / name)
    for folder in ("src", "training", "scripts"):
        shutil.copytree(PROJECT / folder, main / folder, ignore=SKIP)
    docs = main / "docs"
    docs.mkdir()
    available = {name for name in PUBLIC_DOCS if (DOCS / name).exists()}
    write(main / "README.md", front_page(available))
    write(main / "OVERVIEW.md", (PROJECT / "README.md").read_text(encoding="utf-8"))
    pyproject = (main / "pyproject.toml").read_text(encoding="utf-8")
    write(main / "pyproject.toml", pyproject.replace('readme = "README.md"', 'readme = "OVERVIEW.md"'))
    for name in sorted(available - {"README.md"}):
        text = (DOCS / name).read_text(encoding="utf-8")
        if name != "Host Prompt Rules.md":  # kept byte-identical to the packaged copy
            text = convert_markdown(text, available)
        write(docs / name, text)
    shutil.copytree(DOCS / "images", docs / "images")
    if PAPER.exists():
        (docs / "paper").mkdir()
        shutil.copy2(PAPER, docs / "paper" / PAPER.name)
    write(main / "CONTRIBUTING.md", CONTRIBUTING)
    security = (DOCS / "SECURITY.md").read_text(encoding="utf-8")
    security = security.replace(
        "If that is unavailable, email the maintainers at `<security contact email>` with `[mcp-shield security]` in "
        "the subject.",
        "If that is unavailable, open an issue that only asks a maintainer to contact you — no details in public.",
    )
    write(main / "SECURITY.md", security)
    write(main / ".github" / "workflows" / "ci.yml", MAIN_CI)
    write(main / ".github" / "workflows" / "release.yml", RELEASE)
    write(main / ".github" / "CODEOWNERS", CODEOWNERS)
    return main


def stage_tests() -> Path:
    tests = fresh(OUT / "tests")
    shutil.copytree(TESTS, tests, dirs_exist_ok=True, ignore=SKIP)
    write(tests / "README.md", TESTS_README)
    write(tests / ".gitignore", "__pycache__/\n*.pyc\n.pytest_cache/\n.ruff_cache/\n")
    write(tests / ".github" / "workflows" / "tests.yml", TESTS_CI)
    return tests


def stage_dataset() -> Path:
    dataset = fresh(OUT / "dataset")
    guide = (PROJECT / "training" / "CONTRIBUTING-DATA.md").read_text(encoding="utf-8")
    guide = guide.replace(
        "From the main branch's `training/` folder:", "From the `training/` folder of a `main` checkout:"
    )
    write(dataset / "README.md", guide)
    write(dataset / "contributions" / "README.md", CONTRIBUTIONS_README)
    write(dataset / ".gitattributes", "*.jsonl text eol=lf\n")
    write(dataset / ".github" / "workflows" / "check-contributions.yml", DATASET_CI)
    return dataset


def main() -> int:
    OUT.mkdir(exist_ok=True)
    staged = [stage_main(), stage_tests(), stage_dataset()]
    write(OUT / "GIT_COMMANDS.md", GIT_COMMANDS)
    for path in staged:
        files = [p for p in path.rglob("*") if p.is_file()]
        print(f"{path.name:8} {len(files):4} files  {sum(p.stat().st_size for p in files) / 1e6:6.1f} MB")
    print(f"commands: {OUT / 'GIT_COMMANDS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
