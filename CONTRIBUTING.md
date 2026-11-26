# Contributing to MCP-Shield

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
git clone https://github.com/Dummie-Labs/MCP-Shield.git && cd MCP-Shield
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
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
