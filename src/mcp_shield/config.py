"""Configuration: ``SecurityConfig`` (02 §7) and its ``mcp-shield.toml`` form.

The four canonical fields are ``classifier_model``, ``taint_policy``, ``fpr_target``
and ``explainer``; everything else has a safe default. Example file::

    taint_policy = "standard"

    [taint]
    allow_flows = ["filesystem -> github"]

    [tools.shell-tools.run_shell]
    mode = "shell"

    [tools.filesystem.read_file]
    roots = ["~/workspace"]

    [[listeners]]
    name = "postgres"
    listen = "127.0.0.1:8801"
    upstream = "http://localhost:3000/mcp"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

PRIVILEGE_RANK = {"read": 0, "write": 1, "egress": 2, "execute": 3}
TAINT_POLICIES = ("standard", "strict")
TOOL_MODES = ("default", "shell")
FIELD_TYPES = ("path", "command", "url", "text")


class ConfigError(ValueError):
    """Raised for a configuration value that cannot be honoured (fail closed)."""


class ExplainerProvider(str, Enum):
    """Stage 2 Tier 4 explainer providers; ``NONE`` is the default and nothing leaves the machine."""

    NONE = "none"
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"


@dataclass
class ToolPolicy:
    privilege: str | None = None
    mode: str = "default"
    requires_approval: bool = False
    roots: list[str] = field(default_factory=list)
    url_schemes: list[str] = field(default_factory=lambda: ["https", "http"])
    url_hosts: list[str] = field(default_factory=list)
    result_cap: int | None = None
    spotlight: bool = True
    taint_opt_out: bool = False
    field_types: dict[str, str] = field(default_factory=dict)


@dataclass
class ServerPolicy:
    remote: bool | None = None
    spotlight: bool | None = None


@dataclass
class ListenerConfig:
    name: str
    upstream: str
    listen: str = "127.0.0.1:8800"
    bearer_token: str | None = None

    @property
    def host(self) -> str:
        return self.listen.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.listen.rsplit(":", 1)[1])


@dataclass
class SecurityConfig:
    classifier_model: str = "mcp-shield-clf"
    taint_policy: str = "standard"
    fpr_target: float = 0.005
    explainer: ExplainerProvider = ExplainerProvider.NONE

    fail_open: bool = False
    spotlight: bool = True
    model_dir: str | None = None

    idle_window_s: float = 1800.0
    taint_half_life_s: float = 900.0
    taint_live_weight: float = 0.25
    overlap_min_run: int = 32
    cosine_threshold: float = 0.9
    min_free_text: int = 24

    probe_strict_blocks: int = 3
    probe_strict_window_s: float = 600.0
    probe_quarantine_blocks: int = 10
    probe_quarantine_window_s: float = 3600.0

    result_cap_read: int = 2 * 1024 * 1024
    result_cap_default: int = 256 * 1024
    dos_min_size: int = 64 * 1024
    dos_repetition_ratio: float = 20.0

    grant_ttl_s: float = 300.0
    flow_grant_ttl_s: float = 86400.0
    content_allow_ttl_s: float = 86400.0
    cache_ttl_s: float = 86400.0
    retention_days: int = 7

    allow_flows: list[tuple[str, str]] = field(default_factory=list)
    tools: dict[str, dict[str, ToolPolicy]] = field(default_factory=dict)
    servers: dict[str, ServerPolicy] = field(default_factory=dict)
    listeners: list[ListenerConfig] = field(default_factory=list)
    origin_allowlist: list[str] = field(default_factory=list)
    dashboard_port: int = 8501
    explainer_url: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.taint_policy not in TAINT_POLICIES:
            raise ConfigError(f"taint_policy must be one of {TAINT_POLICIES}, got {self.taint_policy!r}")
        if not 0.0 < self.fpr_target < 0.5:
            raise ConfigError(f"fpr_target must be between 0 and 0.5, got {self.fpr_target!r}")
        if isinstance(self.explainer, str):
            self.explainer = ExplainerProvider(self.explainer.lower())

    def tool_policy(self, server: str, tool: str) -> ToolPolicy:
        for key in (server, "*"):
            policy = self.tools.get(key, {}).get(tool)
            if policy is not None:
                return policy
        return ToolPolicy()

    def flow_allowed(self, source: str, destination: str) -> bool:
        return any(src in (source, "*") and dst in (destination, "*") for src, dst in self.allow_flows)


# --------------------------------------------------------------------------- loading


def load_config(path: Path | None = None) -> SecurityConfig:
    """Load ``mcp-shield.toml``; a missing file means defaults. Bad values raise ``ConfigError``."""
    from .paths import config_path

    path = path or config_path()
    if not path.exists():
        return SecurityConfig()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> SecurityConfig:
    raw = dict(raw)
    warnings: list[str] = []
    simple = {f.name for f in fields(SecurityConfig)} - {
        "allow_flows",
        "tools",
        "servers",
        "listeners",
        "origin_allowlist",
        "warnings",
        "explainer",
    }
    kwargs: dict[str, Any] = {}

    shield = raw.pop("shield", {})
    for source in (raw, shield):
        for key in list(source):
            if key in simple:
                kwargs[key] = source.pop(key)

    taint = raw.pop("taint", {})
    if "policy" in taint:
        kwargs["taint_policy"] = taint.pop("policy")
    flows = [_parse_flow(item) for item in taint.pop("allow_flows", [])]
    _warn_unknown(taint, "taint", warnings)

    explainer = raw.pop("explainer", {})
    if "provider" in explainer:
        kwargs["explainer"] = ExplainerProvider(str(explainer.pop("provider")).lower())
    if "url" in explainer:
        kwargs["explainer_url"] = str(explainer.pop("url"))
    _warn_unknown(explainer, "explainer", warnings)

    dashboard = raw.pop("dashboard", {})
    if "port" in dashboard:
        kwargs["dashboard_port"] = int(dashboard.pop("port"))
    origins = list(dashboard.pop("origin_allowlist", []))
    _warn_unknown(dashboard, "dashboard", warnings)

    tools: dict[str, dict[str, ToolPolicy]] = {}
    for server, table in raw.pop("tools", {}).items():
        if not isinstance(table, dict):
            raise ConfigError(f"[tools.{server}] must be a table of tools")
        tools[server] = {tool: _tool_policy(f"{server}.{tool}", spec, warnings) for tool, spec in table.items()}

    servers = {name: _server_policy(name, spec, warnings) for name, spec in raw.pop("servers", {}).items()}
    listeners = [_listener(spec) for spec in raw.pop("listeners", [])]
    for spec in raw.pop("audit", {}), raw.pop("auth", {}), raw.pop("policies", {}), raw.pop("clients", {}):
        _ = spec  # Stage 1E sections: accepted, interpreted by their own modules
    _warn_unknown(raw, "top level", warnings)

    return SecurityConfig(
        **kwargs,
        allow_flows=flows,
        tools=tools,
        servers=servers,
        listeners=listeners,
        origin_allowlist=origins,
        warnings=warnings,
    )


def _parse_flow(item: Any) -> tuple[str, str]:
    if not isinstance(item, str) or "->" not in item:
        raise ConfigError(f"allow_flows entries look like 'source -> destination', got {item!r}")
    src, dst = (part.strip() for part in item.split("->", 1))
    if not src or not dst:
        raise ConfigError(f"allow_flows entry has an empty side: {item!r}")
    return src, dst


def _tool_policy(where: str, spec: Any, warnings: list[str]) -> ToolPolicy:
    if not isinstance(spec, dict):
        raise ConfigError(f"[tools.{where}] must be a table")
    spec = dict(spec)
    policy = ToolPolicy()
    if "privilege" in spec:
        privilege = str(spec.pop("privilege"))
        if privilege not in PRIVILEGE_RANK:
            raise ConfigError(f"[tools.{where}] privilege must be one of {tuple(PRIVILEGE_RANK)}")
        policy.privilege = privilege
    if "mode" in spec:
        mode = str(spec.pop("mode"))
        if mode not in TOOL_MODES:
            raise ConfigError(f"[tools.{where}] mode must be one of {TOOL_MODES}")
        policy.mode = mode
    for key in ("requires_approval", "spotlight", "taint_opt_out"):
        if key in spec:
            setattr(policy, key, bool(spec.pop(key)))
    for key in ("roots", "url_schemes", "url_hosts"):
        if key in spec:
            setattr(policy, key, [str(v) for v in spec.pop(key)])
    if "result_cap" in spec:
        policy.result_cap = int(spec.pop("result_cap"))
    if "field_types" in spec:
        types = {str(k): str(v) for k, v in dict(spec.pop("field_types")).items()}
        bad = {v for v in types.values() if v not in FIELD_TYPES}
        if bad:
            raise ConfigError(f"[tools.{where}] field_types values must be in {FIELD_TYPES}, got {sorted(bad)}")
        policy.field_types = types
    _warn_unknown(spec, f"tools.{where}", warnings)
    return policy


def _server_policy(name: str, spec: Any, warnings: list[str]) -> ServerPolicy:
    spec = dict(spec)
    policy = ServerPolicy()
    if "remote" in spec:
        policy.remote = bool(spec.pop("remote"))
    if "spotlight" in spec:
        policy.spotlight = bool(spec.pop("spotlight"))
    _warn_unknown(spec, f"servers.{name}", warnings)
    return policy


def _listener(spec: Any) -> ListenerConfig:
    try:
        listener = ListenerConfig(
            name=str(spec["name"]),
            upstream=str(spec["upstream"]),
            listen=str(spec.get("listen", "127.0.0.1:8800")),
            bearer_token=spec.get("bearer_token"),
        )
        _ = listener.port
    except (KeyError, ValueError, TypeError) as exc:
        raise ConfigError(f"[[listeners]] entry needs name, upstream and listen = 'host:port': {spec!r}") from exc
    if listener.host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError(f"listener {listener.name!r} must bind to 127.0.0.1, not {listener.host!r}")
    return listener


def _warn_unknown(table: dict[str, Any], where: str, warnings: list[str]) -> None:
    for key in table:
        warnings.append(f"unknown setting '{key}' in [{where}] ignored")
