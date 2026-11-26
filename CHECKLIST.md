# MCP-Shield — build checklist

Tick items in a pull request when they are done **and tested**. Stage tags follow
`docs/02` §9: **S1** Stage 1 core · **1E** enterprise basics (built after Stage 1 is
benchmarked; cut E4 → E3 → E2 if time runs short) · **S2** next semester.

Last updated: 26 Nov 2026 (Phase A1–A3 done).

## Done (tested on Windows, macOS and Linux in CI)

- [x] Engine (one per OS user), local channel (Unix socket / owner-only named pipe, mutual HMAC), autostart
- [x] stdio front-end (`mcp-shield proxy --name … -- …`), Windows Job Object, fail-closed when the engine is unreachable
- [x] Streamable HTTP listener: header allowlist before the body, header ↔ body (`-32020`), SSE inspection, Host / Origin / bearer checks
- [x] Tier 1 (shape, schema, per-OS argument policy, privilege classes), Tier 2 (cache, signatures, DoS guard), Tier 3 runtime (ONNX, chunking, thresholds, abstain band, fail-closed)
- [x] Untrusted-segment extractor, Spotlighting, MRTR / legacy provenance banners and declines
- [x] Capability pinning, description scan, first-contact quarantine
- [x] Session taint: winnowed fingerprints, containment, write-back exemption, trusted flows, decay, persistence
- [x] Strict policy, approve-then-retry (*Allow once*, *Allow this flow*), probe detection and quarantine
- [x] Opaque block stub + 32-bit reference, SQLite event store (redacted, 7-day retention)
- [x] Dashboard: traffic log, flagged-request inspector with one-click actions, pin viewer, sessions
- [x] CLI: `doctor`, `install` / `uninstall`, `rules`, `proxy`, `engine`, `dashboard`, `pull-model`, `pin`, `events`, `action`, `benchmark` (latency, replay)
- [x] Host prompt rules v0.4 (H-1 … H-25) with an integrity test
- [x] Training pipeline: contributions → manifest → ingest (held-out guard, dedup, grouped split, augmentation) → train → export (ONNX int8 gate) → calibrate → evaluate
- [x] CI (3 OS × Python 3.10 / 3.13), release workflow, rulesets, CODEOWNERS

## Phase A — the whole package working with an interim classifier (now)

- [x] **A1** Interim classifier `mcp-shield pull-model --interim`: `protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0) at a pinned revision with pinned SHA-256s, fp32, self-checked after install; `doctor` and the dashboard label it *interim*
  - Known weaknesses (measured 26 Nov on a small check set): blocks text that quotes injection phrases, ordinary "follow the instructions in the email" sentences, and credential-looking text such as `.env` contents; misses command injection inside result text; English only. Fine-tuning on our hard negatives (Phase C) is the fix.
- [x] **A2** Runtime support: two-label models, models trained without the context prefix, fail-closed on a label/output mismatch
- [x] **A3** `doctor` self-test and `benchmark --suite latency` on the interim model, CPU fp32, Windows dev PC (26 Nov): Tier 1 0.01 ms; Tier 3 p50 / p95 — short 17 / 18 ms, ~512 tokens 123 / 134 ms, ~3,000 words 1.5 s; engine path for a ~200-word result 57 / 60 ms. Linux numbers come with D4.
- [ ] **A4** Real hosts, by hand: Claude Desktop (Windows, macOS) and Cursor (all three) — `install`, first-contact quarantine, accept pins, reference servers (filesystem, GitHub), a planted injection blocked, a taint flow held, dashboard *Allow once*
- [ ] **A5** Fix what A3 / A4 find
- [ ] **A6** First pre-release: tag `v0.1.0.dev1` → GitHub release with the wheel

## Phase B — data (team, from 27–28 Nov)

- [ ] **B1** Each member adds `contributions/<name>-<NNN>.jsonl` on the `dataset` branch (guide: `training/CONTRIBUTING-DATA.md`)
- [ ] **B2** Benign capture from sandbox accounts through the reference servers, with ≥ 25 % hard negatives
- [ ] **B3** Public attack corpora (InjecAgent, …) converted, licences recorded
- [ ] **B4** Held-out files (AgentDojo, MCPSecBench, red-team set) kept outside training and listed as `[[heldout]]`
- [ ] **B5** ≥ 20,000 segments and the class mix of `docs/03` §6 (the dataset card reports both)

## Phase C — train `mcp-shield-clf`

- [ ] **C1** Training environment: CUDA build of PyTorch on the RTX 5060 Ti
- [ ] **C2** Base weights `microsoft/mdeberta-v3-base`
- [ ] **C3** `ingest` → `train` (3 seeds) → `export` → `evaluate`
- [ ] **C3b** Quantisation strategy: dynamic int8 broke the interim DeBERTa-v3 model (scores moved by up to 0.99 in every variant tried, feed-forward-only included), and `mcp-shield-clf` is the same family. Try static int8 with calibration data, or fp16 embeddings; otherwise ship fp32 (≈ 1.1 GB for mDeBERTa) and update the footprint figures in the docs
- [ ] **C4** Adversarial rounds (`docs/03` §7.1) — attack-transform and validity-filter tooling still to write
- [ ] **C5** Leave-one-source-out results; per-document FPR on long documents
- [ ] **C6** Publish the weights on Hugging Face; pin revision + SHA-256 in `models.py`; model card with the measured numbers
- [ ] **C7** Replace the interim model with `mcp-shield-clf`

## Phase D — benchmarks (S1)

- [x] Latency and replay suites
- [ ] **D1** AgentDojo and MCPSecBench adapters (held-out)
- [ ] **D2** Agent under attack: `llama-server` + Qwen3-8B GGUF with `--jinja`
- [ ] **D3** Prompt Guard 2 baseline on the same sets
- [ ] **D4** CPU latency baseline job in CI (4-vCPU runner)
- [ ] **D5** Utility retained, with the cross-server taint rule on and off

## Phase R — release (S1)

- [x] GitHub release workflow (tag → wheel + sdist)
- [ ] **R1** PyPI Trusted Publishing for `mcp-shield-llm`
- [ ] **R2** Known manifests for the reference servers (`scripts/pin_reference_servers.py`)

## Stage 1E — enterprise basics

- [ ] **E1** JSONL audit log + OTLP export with trace-context propagation
- [ ] **E2** Dockerfile, docker-compose, Kubernetes sidecar manifest
- [ ] **E3** OAuth 2.1 bearer validation (JWKS, `iss` / `aud` / `exp`) + policy map
- [ ] **E4** Per-client config sections and allowlists

## Small gaps in the current code

- [ ] List results honour `ttlMs` but not `cacheScope`
- [ ] `resources/read` is not checked against withheld resource templates
- [ ] `subscriptions/listen` → `toolsListChanged` passes through; the re-check happens on the host's next list
- [ ] `docs/Host Prompt Rules.md` keeps one Obsidian `[[link]]` (shown literally on GitHub)

## Stage 2 (next semester)

`@shield` decorator + ASGI middleware · drift / loop-breaker state tracker · Tier 4 explainer
(`llama-server`) · in-host approval via MRTR elicitation · full dashboard (topology, tool hub,
model switcher) · OAuth proxy pattern for third-party remote servers · gated + rotated weights.
