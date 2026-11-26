# MCP-Shield

A semantic firewall for the Model Context Protocol. MCP-Shield runs on your machine,
between an MCP host (Claude Desktop, Cursor, a custom agent) and the MCP servers it
uses, and inspects every message in both directions:

- **Tier 1** — structural and policy checks: JSON-RPC shape, `Mcp-*` header ↔ body
  consistency, the pinned input schema, and an argument policy for paths, commands and
  URLs (per-OS operator sets and path canonicalisation).
- **Tier 2** — an exact-hash cache of content already cleared, a signature set of
  confirmed attacks, and a Context-DoS guard.
- **Tier 3** — a fine-tuned encoder classifier (`mcp-shield-clf`, ONNX Runtime, CPU by
  default) over every untrusted free-text segment, with an abstain band.
- **Session taint** — results that reached the host are fingerprinted; a write, execute
  or egress call, or any call to another remote server, that carries that content is
  blocked or held for approval.
- **Capability pinning** — tool, prompt and template definitions are pinned (SHA-256)
  and scanned; a new server starts quarantined until you review it.
- **Spotlighting** — forwarded untrusted text is wrapped in markers, and a versioned set
  of host prompt rules (`mcp-shield rules --print`) tells the host what they mean.

Blocks are opaque by design: the host receives only
`Request blocked by MCP-Shield policy (ref xxxxxxxx)`; the reason is in the local
dashboard, under that reference.

> Status: Stage 1, pre-release. Latency, accuracy and footprint figures in the design
> documents are **targets** until the benchmark phase measures them.

## Install

```bash
pip install mcp-shield-llm          # Python 3.10 – 3.13 on Windows, macOS or Linux
mcp-shield pull-model --from <dir>  # install classifier weights (see "Classifier weights")
mcp-shield doctor                   # checks the install and picks the fastest execution provider
```

ONNX Runtime ships as mutually exclusive wheels. The package depends on the CPU wheel;
`doctor` prints the one-line swap when it finds a CUDA or DirectML GPU and keeps an
accelerator only if it beats the CPU on its self-test.

## Use

```bash
mcp-shield install --host claude-desktop   # or: cursor, or a path to an MCP config JSON
mcp-shield install --host cursor --with-rules
mcp-shield dashboard                        # prints a one-time login link to 127.0.0.1:8501
```

`install` rewrites the host's config so each stdio server runs as
`mcp-shield proxy --name <server> -- <original command>` and each non-OAuth HTTP server
goes through an engine listener. It keeps a backup, shows the full plan first, and
`mcp-shield uninstall --host …` restores exactly the entries it changed. Servers that
probably use OAuth are reported and left untouched.

New servers start **quarantined**: only read-class tools are exposed until you accept
the manifest.

```bash
mcp-shield pin --server github --show
mcp-shield pin --server github --accept
```

### Commands

| Command | What it does |
|:--|:--|
| `doctor` | OS, event loop, ONNX Runtime providers (warns on duplicate wheels), paths, model checksums, self-test, unproxied servers |
| `install` / `uninstall` | route a host's servers through MCP-Shield, or restore them |
| `rules --print [--short]` | the host prompt rules (H-1 … H-24) |
| `proxy --name N -- CMD…` | a stdio front-end (what `install` writes into the host config) |
| `proxy --listen 127.0.0.1:P --upstream URL` | ask the engine to open an HTTP listener |
| `engine [--status \| --foreground \| --stop]` | the per-user engine (normally started by the first front-end) |
| `dashboard` | open the local dashboard |
| `pull-model [--from DIR]` | install and verify classifier weights |
| `pin --server S [--show \| --accept \| --forget]` | capability pins |
| `events [--ref R]` / `action --ref R --do …` | the event store and the dashboard's actions, from a terminal |
| `benchmark --suite latency \| replay` | latency per tier; FPR / recall on a labelled JSONL |

### Configuration

`mcp-shield.toml` in the per-user config directory (`doctor` prints the path):

```toml
taint_policy = "standard"            # or "strict": execute/egress calls in a tainted session need approval

[taint]
allow_flows = ["filesystem -> github"]

[tools.shell-tools.run_shell]
mode = "shell"                       # shell operators allowed; every call held once the session is tainted

[tools.filesystem.read_file]
roots = ["~/workspace"]

[[listeners]]
name = "postgres"
listen = "127.0.0.1:8801"
upstream = "http://localhost:3000/mcp"
```

## How it runs

One **engine** per OS user holds the classifier, the cache, the taint store, the pin
store and a SQLite event store, and serves the dashboard. Every `mcp-shield proxy` is a
thin front-end that talks to it over a Unix socket (macOS, Linux) or an owner-only named
pipe (Windows), after a mutual HMAC challenge on a per-user secret. HTTP listeners run
inside the engine. The engine never starts an MCP server. If it cannot be reached, the
front-end blocks traffic in both directions (fail closed).

## Classifier weights

The weights are not in the package. Until `mcp-shield-clf` is published, install the
**interim** classifier — an off-the-shelf prompt-injection model, pinned and self-checked —
or a model you have built:

```bash
mcp-shield pull-model --interim                # ~750 MB download; English only, a stand-in
mcp-shield pull-model --from path/to/model     # model.onnx, tokenizer.json, mcp_shield_clf.json
```

The interim model is labelled as such by `doctor` and the dashboard. It blocks some harmless
text (quoted injection phrases, credential-looking content) and misses command injection in
result text; the fine-tuned `mcp-shield-clf` replaces it.

Without a model, the engine fails closed: untrusted content is blocked and tool
descriptions are withheld. `--fail-open` exists for development only.

## Training the classifier

The pipeline lives in `training/` (install with `pip install -e ".[train]"`) and runs
from that directory:

```bash
python -m mcp_shield_train ingest   --manifest dataset.toml          # collected files → dataset + card
python -m mcp_shield_train check    --data data/v1
python -m mcp_shield_train train    --data data/v1 --out runs/r1
python -m mcp_shield_train export   --run runs/r1 --data data/v1 --out models/v1 --version 0.1.0
python -m mcp_shield_train evaluate --model models/v1 --data heldout.jsonl
mcp-shield pull-model --from models/v1
```

`dataset.example.toml` shows how to describe each source file (text column, label or
label mapping, group id, context prefix, hard negatives, licence) and the held-out
benchmark files whose texts must never enter training. `ingest` applies the order the
design requires: held-out guard, deduplication within each label, grouped stratified
80/10/10 split, augmentation on train only. `export` writes the ONNX graph with logits
and pooled outputs, quantises it to int8 (embedding table included) behind a recall gate,
and calibrates temperature and the length-adjusted thresholds on the shipped graph.

## Documentation

The design overview is the repository's front page ([`README.md`](README.md)); the full design is in
[`docs/`](docs/): the threat model, the canonical architecture
([`02`](docs/02%20-%20System%20Architecture%20%26%20Technical%20Implementation.md)), the
dataset and training strategy, the host prompt rules, and the roadmap. The research paper
is [`docs/paper/MCP-Shield IEEE Paper.pdf`](docs/paper/MCP-Shield%20IEEE%20Paper.pdf).

## Branches

| Branch | What it holds |
|:--|:--|
| `main` | this code, the docs and the paper |
| `tests` | the test suite (CI checks it out next to `main` on Windows, macOS and Linux) |
| `dataset` | data contributions for the classifier — see `training/CONTRIBUTING-DATA.md` |

## Development

```bash
git clone https://github.com/DUME-Labs/MCP-Shield.git && cd MCP-Shield
pip install -e ".[dev]"
git worktree add ../MCP-Shield-tests tests          # the suite, beside the code
cd ../MCP-Shield-tests
MCP_SHIELD_PROJECT=../MCP-Shield python -m pytest   # PowerShell: $env:MCP_SHIELD_PROJECT="..\MCP-Shield"
python -m pytest --run-training-code                # also check the train/export path on a toy model
```

Lint and types: `ruff check src training scripts` and `mypy src`.

## References

MCP-Shield builds on the analysis in: N. Maloyan and D. Namiot, "Breaking the Protocol:
Security Analysis of the Model Context Protocol Specification and Prompt Injection
Vulnerabilities in Tool-Integrated LLM Agents," arXiv:2601.17549, Jan. 2026 —
<https://arxiv.org/abs/2601.17549>. The full bibliography is in
[`docs/05`](docs/05%20-%20Literature%20Survey%20%26%20Research%20Papers.md).

## Limitations

- Third-party OAuth-protected remote servers cannot be fronted transparently (Stage 2).
- Taint sessions are per host application; two windows of one host share a session.
- Images returned by tools are hashed for taint, not inspected.
- Any server update changes descriptions, so it trips the pin until you re-accept.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.
