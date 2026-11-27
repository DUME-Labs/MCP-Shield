# MCP-Shield website — planned, not built yet

This branch will hold the project website, served by **GitHub Pages** at
<https://dume-labs.github.io/MCP-Shield/>.

**Building starts after the Phase A release** — the first version on PyPI together with the
GitHub release and its wheel (`CHECKLIST.md` on `main`: A6, then Phase W). Until then this
branch only holds this plan.

## What the site will do

1. **Say what MCP-Shield is in one screen**: a local firewall between an AI app (Claude
   Desktop, Cursor, an agent) and its MCP servers — what it stops (indirect prompt
   injection, poisoned tool descriptions, data leaking from one server to another, command
   injection) and, just as plainly, what it does not.
2. **Give the right install commands for the visitor's OS**, detected automatically, with
   tabs to switch by hand (detection can be wrong) and a copy button on each command:
   - Windows: `py -m pip install mcp-shield-llm`
   - macOS / Linux: `python3 -m pip install mcp-shield-llm` (or `pipx install mcp-shield-llm`)
   - then `mcp-shield pull-model …`, `mcp-shield doctor`,
     `mcp-shield install --host claude-desktop` or `--host cursor`
   - the one-line ONNX Runtime swap for an NVIDIA or DirectX 12 GPU, as `doctor` prints it
3. **Showcase the project**: the request path through the checks, the dashboard
   (screenshots), the host prompt rules, measured numbers only (with the machine and date),
   the docs, the research paper and the data contribution guide.
4. **Stay current on its own**: version, release notes and the wheel link come from the
   GitHub release and PyPI, so the commands never go out of date.

## How it will be built and deployed

- A static site (HTML, CSS, a little JavaScript; no server), dark-first like the dashboard,
  readable on a phone, keyboard- and screen-reader-friendly.
- OS detection in the browser (`navigator.userAgentData.platform`, falling back to
  `navigator.userAgent`). No tracking, no cookies, no third-party scripts.
- Deployed by a GitHub Actions workflow on this branch (Settings → Pages → Source:
  GitHub Actions).
- To decide at build time: whether the first PyPI version is a pre-release (then the
  commands need `--pre`), a custom domain, and which screenshots to use.
