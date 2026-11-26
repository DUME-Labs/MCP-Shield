// MCP-Shield dashboard. Every value that came from traffic is attacker-controlled:
// it is only ever inserted with textContent, never as HTML.
"use strict";

const state = { flagged: [], selected: null, pinServer: null };
const FLAG_VERDICTS = "block,hold,abstain,flag";

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options) {
  const opts = Object.assign({ credentials: "same-origin", headers: {} }, options || {});
  if (opts.method && opts.method !== "GET") {
    opts.headers["X-MCP-Shield"] = "1";
    opts.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(text) {
  const t = document.getElementById("toast");
  t.textContent = text;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 4000);
}

const time = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "");
const verdict = (v) => el("span", { class: "v " + (v || "info") }, v || "");
const ms = (t, key) => (t && t[key] !== undefined ? t[key].toFixed(1) + " ms" : "");

// ------------------------------------------------------------------ tabs + theme

function showTab(name) {
  document.querySelectorAll(".side button").forEach((b) => {
    if (b.dataset.tab === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  document.querySelectorAll(".tab").forEach((s) => (s.dataset.active = String(s.id === "tab-" + name)));
  if (name === "pins") loadServers();
  if (name === "sessions") loadSessions();
  if (name === "engine") loadStatus();
  if (name === "flagged") loadFlagged();
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("mcp-shield-theme"); } catch (e) { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  document.getElementById("theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("mcp-shield-theme", next); } catch (e) { /* ignore */ }
  });
}

// ------------------------------------------------------------------ traffic

function trafficRow(e) {
  const t = e.timings || {};
  return el("tr", {}, el("td", {}, time(e.ts)), el("td", {}, verdict(e.verdict)), el("td", {}, e.kind || ""),
    el("td", {}, e.server || ""), el("td", { class: "mono" }, e.tool || e.method || ""),
    el("td", {}, ms(t, "tier2_ms")), el("td", {}, ms(t, "tier3_ms")), el("td", { class: "mono" }, e.ref || ""));
}

async function loadTraffic() {
  const rows = await api("/api/events?limit=200");
  const body = document.getElementById("traffic-body");
  body.replaceChildren(...rows.map(trafficRow));
}

function prependTraffic(e) {
  const body = document.getElementById("traffic-body");
  body.prepend(trafficRow(e));
  while (body.children.length > 300) body.lastChild.remove();
}

// ------------------------------------------------------------------ flagged

async function loadFlagged() {
  const q = document.getElementById("flag-search").value.trim();
  const rows = await api("/api/events?limit=300&verdict=" + FLAG_VERDICTS + (q ? "&q=" + encodeURIComponent(q) : ""));
  state.flagged = rows;
  const open = rows.filter((r) => r.status === "open" && (r.verdict === "block" || r.verdict === "hold")).length;
  document.getElementById("flag-count").textContent = open ? String(open) : "";
  const body = document.getElementById("flag-body");
  body.replaceChildren(...rows.map((e) => {
    const tr = el("tr", { class: "clickable" + (state.selected === e.ref ? " selected" : "") },
      el("td", {}, time(e.ts)), el("td", {}, verdict(e.verdict)), el("td", {}, e.kind || ""),
      el("td", {}, e.server || ""), el("td", { class: "mono" }, e.tool || e.method || ""),
      el("td", { class: "mono" }, e.ref));
    tr.addEventListener("click", () => selectFlagged(e.ref));
    return tr;
  }));
}

const ACTION_LABELS = {
  allow_content: "Allow this exact content", allow_once: "Allow once", allow_flow: "Allow this flow (24 h)",
  raise_cap: "Raise this tool's result cap", mark_benign: "Mark benign", mark_attack: "Mark as attack",
  report_fp: "Report false positive", accept_pins: "Review pins",
};

function actionsFor(e) {
  if (e.status !== "open") return [];
  if (e.origin === "server") return [];
  const out = [];
  if (e.kind === "injection" && e.segment_sha) out.push("allow_content");
  if (e.kind === "taint") out.push("allow_once", "allow_flow");
  if (e.kind === "hold") out.push("allow_once");
  if (e.kind === "dos" || e.kind === "truncate") out.push("raise_cap");
  if (e.kind === "abstain") out.push("mark_benign", "mark_attack");
  if ((e.tier === "tier3" || e.tier === "tier2") && (e.kind === "injection" || e.kind === "abstain")) out.push("report_fp");
  if (e.kind === "tamper" || e.kind === "quarantine") out.push("accept_pins");
  return out;
}

async function selectFlagged(ref) {
  state.selected = ref;
  const e = await api("/api/events/" + encodeURIComponent(ref));
  const pane = document.getElementById("flag-detail");
  const dl = el("dl", {});
  const add = (k, v) => { if (v !== null && v !== undefined && v !== "") dl.append(el("dt", {}, k), el("dd", {}, v)); };
  add("Reference", e.ref); add("Verdict", e.verdict); add("Kind", e.kind); add("Tier", e.tier);
  add("Class", e.cls); add("Probability", typeof e.prob === "number" ? e.prob.toFixed(3) : "");
  add("Server", e.server); add("Tool", e.tool); add("Direction", e.direction); add("Session", e.session);
  add("Trace", e.trace_id); add("Origin", e.origin === "server" ? "returned by the server (not a proxy block)" : "MCP-Shield");
  add("Reason", e.reason); add("Sources", (e.sources || []).join(", ")); add("Status", e.status);
  const children = [el("h2", {}, "Why"), dl];
  if (e.segment) children.push(el("h3", {}, "Segment (local, redacted)"), el("pre", { class: "segment" }, e.segment));
  if (e.diff && e.diff.length) children.push(el("h3", {}, "Definition diff"), el("pre", { class: "segment" }, JSON.stringify(e.diff, null, 2)));
  const actions = actionsFor(e);
  if (actions.length) {
    const box = el("div", {});
    for (const a of actions) {
      box.append(el("button", { class: "act", type: "button", onclick: () => runAction(e, a) }, ACTION_LABELS[a]));
    }
    children.push(el("h3", {}, "Actions"), box);
  } else if (e.kind === "policy") {
    children.push(el("p", { class: "muted" }, "Tier 1 policy blocks have no one-click override: fix the policy (roots, field types, mode = \"shell\") in mcp-shield.toml."));
  }
  pane.replaceChildren(...children);
  loadFlagged();
}

async function runAction(e, action) {
  if (action === "accept_pins") { state.pinServer = e.server; showTab("pins"); return; }
  if (!confirm(ACTION_LABELS[action] + " for " + e.ref + "?")) return;
  try {
    const res = await api("/api/events/" + encodeURIComponent(e.ref) + "/action", { method: "POST", body: JSON.stringify({ action }) });
    toast(res.note || "done");
    selectFlagged(e.ref);
  } catch (err) { toast("Failed: " + err.message); }
}

// ------------------------------------------------------------------ pins

async function loadServers() {
  const servers = await api("/api/servers");
  const sel = document.getElementById("pin-server");
  sel.replaceChildren(...servers.map((s) => el("option", { value: s.name }, s.name + " (" + s.status + ")")));
  if (state.pinServer && servers.some((s) => s.name === state.pinServer)) sel.value = state.pinServer;
  if (sel.value) loadPins(sel.value);
}

async function loadPins(server) {
  state.pinServer = server;
  const data = await api("/api/pins/" + encodeURIComponent(server));
  const s = data.server || {};
  document.getElementById("pin-status").textContent = "Status: " + (s.status || "unknown") + (s.status_reason ? " — " + s.status_reason : "") + (s.command ? " · command: " + s.command : "");
  const body = document.getElementById("pin-body");
  body.replaceChildren(...data.items.map((p) => el("tr", {},
    el("td", {}, p.kind), el("td", { class: "mono" }, p.name), el("td", {}, p.privilege || ""),
    el("td", {}, p.changed ? "changed" : (p.accepted ? "accepted" : "pending review")),
    el("td", {}, (p.scan.decision || "") + (p.scan.label && p.scan.label !== "benign" ? " · " + p.scan.label : "")),
    el("td", {}, p.exposed ? "yes" : "no — " + (p.reason || "")))));
  const diffs = data.items.filter((p) => p.diff && p.diff.length);
  const box = document.getElementById("pin-diff");
  box.replaceChildren(...diffs.map((p) => el("div", {}, el("h3", {}, "Changed: " + p.name), el("pre", { class: "segment" }, JSON.stringify(p.diff, null, 2)))));
}

async function acceptPins() {
  const server = document.getElementById("pin-server").value;
  if (!server) return;
  if (!confirm("Accept every current definition of '" + server + "' after reviewing the descriptions and scan verdicts? This exposes its write, execute and egress tools.")) return;
  try {
    const res = await api("/api/pins/" + encodeURIComponent(server) + "/accept", { method: "POST", body: JSON.stringify({ confirm: true }) });
    toast(res.accepted + " definitions accepted");
    loadPins(server);
  } catch (err) { toast("Failed: " + err.message); }
}

async function releaseServer() {
  const server = document.getElementById("pin-server").value;
  if (!server || !confirm("Release the probe quarantine of '" + server + "'?")) return;
  await api("/api/servers/" + encodeURIComponent(server) + "/release", { method: "POST", body: "{}" });
  toast("released");
  loadPins(server);
}

// ------------------------------------------------------------------ sessions + engine

async function loadSessions() {
  const rows = await api("/api/sessions");
  document.getElementById("session-body").replaceChildren(...rows.map((s) => el("tr", {},
    el("td", {}, s.client || "(unknown)"), el("td", { class: "mono" }, s.host), el("td", {}, s.policy + (s.quarantined ? " · quarantined" : "") + (s.strict_reasons.length ? " (" + s.strict_reasons.join(", ") + ")" : "")),
    el("td", {}, s.tainted ? "yes (" + s.live_records + ")" : "no"), el("td", {}, s.sources.join(", ")), el("td", {}, time(s.last_seen)),
    el("td", {}, s.policy === "strict" || s.quarantined ? el("button", { class: "act", type: "button", onclick: async () => {
      if (!confirm("Reset this session to standard policy?")) return;
      await api("/api/sessions/" + encodeURIComponent(s.key) + "/reset", { method: "POST", body: "{}" });
      toast("session reset"); loadSessions();
    } }, "Reset to standard") : ""))));
}

async function loadStatus() {
  const s = await api("/api/status");
  document.getElementById("version").textContent = "v" + s.version;
  document.getElementById("engine-status").textContent = JSON.stringify(s, null, 2);
  const pills = [];
  pills.push(el("span", { class: "pill " + (s.classifier.available ? "good" : "crit") }, s.classifier.available ? "classifier " + s.classifier.version : "classifier unavailable — failing closed"));
  if (s.fail_open) pills.push(el("span", { class: "pill crit" }, "FAIL-OPEN (development)"));
  pills.push(el("span", { class: "pill" }, "policy " + s.taint_policy));
  pills.push(el("span", { class: "pill" }, s.frontends.length + " front-ends"));
  document.getElementById("pills").replaceChildren(...pills);
}

// ------------------------------------------------------------------ live feed

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  const live = document.getElementById("live");
  ws.onopen = () => { live.textContent = "live"; live.classList.add("on"); };
  ws.onclose = () => { live.textContent = "offline"; live.classList.remove("on"); setTimeout(connect, 3000); };
  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.type !== "event") return;
    prependTraffic(data.event);
    if (FLAG_VERDICTS.split(",").includes(data.event.verdict)) loadFlagged();
  };
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  document.querySelectorAll(".side button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.getElementById("flag-search").addEventListener("input", () => loadFlagged());
  document.getElementById("pin-server").addEventListener("change", (e) => loadPins(e.target.value));
  document.getElementById("pin-accept").addEventListener("click", acceptPins);
  document.getElementById("pin-release").addEventListener("click", releaseServer);
  loadStatus().catch(() => {});
  loadTraffic().catch(() => {});
  loadFlagged().catch(() => {});
  connect();
  setInterval(() => loadStatus().catch(() => {}), 15000);
});
