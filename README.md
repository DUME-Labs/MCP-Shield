# MCP-Shield — test suite

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
