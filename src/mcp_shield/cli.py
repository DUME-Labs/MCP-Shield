"""``mcp-shield`` command line (07 §3)."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, SecurityConfig, load_config


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("not a terminal: re-run with --yes to confirm", file=sys.stderr)
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _engine(cmd: str, *, autostart: bool = False, **kwargs: Any) -> Any:
    from .engine.client import call_engine
    from .engine.server import new_event_loop

    loop = new_event_loop()
    try:
        return loop.run_until_complete(call_engine(cmd, autostart=autostart, **kwargs))
    finally:
        loop.close()


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- commands


def cmd_doctor(args: argparse.Namespace, config: SecurityConfig) -> int:
    from .doctor import run

    report = run(config, do_self_test=not args.no_self_test)
    print(report.render())
    return 1 if report.problems else 0


def cmd_install(args: argparse.Namespace, config: SecurityConfig) -> int:
    from . import install, rules
    from .engine import autostart

    path = install.resolve_host(args.host)
    plan = install.plan_install(path)
    print(f"Host config: {path}")
    if not plan.changes:
        print("No MCP servers found in this config.")
    for change in plan.changes:
        print("  " + change.describe())
    if plan.needs_login_item:
        print(f"  [login] register the engine as a login item: {autostart.register(dry_run=True)}")
    if args.with_rules:
        print("  [rules] insert the MCP-Shield host rules where this host exposes an instructions file")
    if args.dry_run:
        return 0
    if plan.effective and not _confirm("Apply these changes (a backup is kept)?", args.yes):
        print("Nothing changed.")
        return 1
    backup = install.apply_install(plan)
    if backup:
        print(f"Updated {path} (backup: {backup})")
    if plan.needs_login_item:
        print(f"Registered {autostart.register()}")
    if args.with_rules:
        if args.host == "cursor" or path.parent.name == ".cursor":
            print(f"Wrote host rules to {install.write_cursor_rules(path, rules.rules_text())}")
        else:
            print(
                "This host has no instructions file MCP-Shield can write. Paste the following into its "
                "system / project instructions:\n"
            )
            print(rules.rules_text())
    print(
        "Restart the host so it relaunches its servers through MCP-Shield. New servers start quarantined: "
        "review them with `mcp-shield pin --server <name> --show` or in the dashboard."
    )
    return 0


def cmd_uninstall(args: argparse.Namespace, config: SecurityConfig) -> int:
    from . import install
    from .engine import autostart

    path = install.resolve_host(args.host)
    restore = install.plan_uninstall(path)
    for name, _ in restore:
        print(f"  [restore] {name}")
    if not restore:
        print("Nothing to restore (entries were changed since install, or install never ran on this config).")
    if restore and not _confirm("Restore these entries?", args.yes):
        return 1
    restored = install.apply_uninstall(path)
    print(f"Restored {len(restored)} entr{'y' if len(restored) == 1 else 'ies'} in {path}")
    if not install.any_listeners_installed():
        removed = autostart.unregister()
        if removed:
            print(removed)
    return 0


def cmd_rules(args: argparse.Namespace, config: SecurityConfig) -> int:
    from . import rules

    print(rules.short_text() if args.short else rules.rules_text(), end="")
    return 0


def cmd_proxy(args: argparse.Namespace, config: SecurityConfig, command: list[str]) -> int:
    if args.listen or args.upstream:
        if not (args.listen and args.upstream):
            print("--listen and --upstream go together", file=sys.stderr)
            return 2
        name = args.name or "http"
        result = _engine(
            "open_listener",
            autostart=True,
            name=name,
            listen=args.listen,
            upstream=args.upstream,
            bearer_token=args.bearer_token,
        )
        print(f"engine listener '{name}' on {result['listen']} -> {args.upstream}")
        print("Add it to mcp-shield.toml ([[listeners]]) or use `mcp-shield install` so it survives an engine restart.")
        return 0
    if not command:
        print("usage: mcp-shield proxy --name <server> -- <server command…>", file=sys.stderr)
        return 2
    from .engine.server import new_event_loop
    from .proxy.stdio import run_stdio

    name = args.name or Path(command[0]).stem
    return run_stdio(name, command, new_event_loop)


def cmd_engine(args: argparse.Namespace, config: SecurityConfig) -> int:
    from .engine.server import run_engine

    if args.foreground or args.background:
        if args.fail_open:
            config.fail_open = True
        return run_engine(config, foreground=args.foreground)
    if args.stop:
        try:
            _engine("shutdown")
            print("engine stopping")
        except Exception as exc:
            print(f"engine not running ({exc})")
        return 0
    try:
        _print(_engine("status"))
        return 0
    except Exception as exc:
        print(f"engine not running ({exc})")
        return 1


def cmd_dashboard(args: argparse.Namespace, config: SecurityConfig) -> int:
    result = _engine("dashboard_url", autostart=True, port=args.port)
    url = result["url"]
    print(f"Dashboard: {url}\n(one-time link: the page exchanges the token for a session cookie; keep it private)")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


def cmd_pull_model(args: argparse.Namespace, config: SecurityConfig) -> int:
    from . import models

    if args.model:
        config.classifier_model = args.model
    try:
        if args.source:
            target = models.install_from(Path(args.source), config)
        else:
            target = models.pull(config, allow_unpinned=args.allow_unpinned, include_fp16=args.fp16)
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"model installed and verified in {target}")
    print("Restart the engine (`mcp-shield engine --stop`; the next front-end starts it) to load it.")
    return 0


def _pins_local(server: str) -> dict[str, Any]:
    from .cascade.tier3_classifier import UnavailableClassifier
    from .engine.core import Inspector
    from .store import Store

    store = Store()
    insp = Inspector(SecurityConfig(), store, UnavailableClassifier("cli"))
    try:
        return {"server": store.server(server), "items": insp.pins_view(server)}
    finally:
        insp.close()
        store.close()


def cmd_pin(args: argparse.Namespace, config: SecurityConfig) -> int:
    if args.accept:
        view = _pin_view(args.server)
        for item in view["items"]:
            flag = "CHANGED" if item["changed"] else ("accepted" if item["accepted"] else "pending")
            print(
                f"  {item['kind']:8} {item['name']:32} {item.get('privilege') or '':8} {flag:9} "
                f"scan={item['scan']['decision']}"
            )
        if not view["items"]:
            print(f"no definitions recorded for '{args.server}' yet (the server has not been listed)")
            return 1
        if not _confirm(f"Accept every current definition of '{args.server}'?", args.yes):
            return 1
        try:
            result = _engine("accept_pins", server=args.server)
        except Exception:
            from .store import Store

            store = Store()
            result = {"accepted": store.accept_pins(args.server)}
            store.close()
        print(f"accepted {result['accepted']} definitions for '{args.server}'")
        return 0
    if args.forget:
        if not _confirm(f"Forget every pin of '{args.server}' (it will be quarantined again)?", args.yes):
            return 1
        from .store import Store

        store = Store()
        store.forget_server(args.server)
        store.close()
        print(f"forgot '{args.server}'")
        return 0
    view = _pin_view(args.server)
    if args.json:
        _print(view)
        return 0
    server = view.get("server") or {}
    print(f"{args.server}: status {server.get('status', 'unknown')}  command: {server.get('command') or '-'}")
    for item in view["items"]:
        state = "CHANGED" if item["changed"] else ("accepted" if item["accepted"] else "pending review")
        print(
            f"\n  {item['kind']} {item['name']}  [{item.get('privilege') or '-'}]  {state}  exposed={item['exposed']}"
            + (f" ({item['reason']})" if item["reason"] else "")
        )
        print(f"    scan: {item['scan']['decision']} {item['scan']['label'] or ''} {item['scan']['prob'] or ''}")
        if item["description"]:
            print("    description: " + item["description"].replace("\n", "\n                 "))
        for d in item["diff"]:
            print(f"    diff {d['field']}: {json.dumps(d['old'])[:300]} -> {json.dumps(d['new'])[:300]}")
    return 0


def _pin_view(server: str) -> dict[str, Any]:
    try:
        return dict(_engine("pins", server=server))
    except Exception:
        return _pins_local(server)


def cmd_events(args: argparse.Namespace, config: SecurityConfig) -> int:
    from .store import Store

    store = Store()
    try:
        rows = store.events(args.limit, query=args.ref)
    finally:
        store.close()
    for row in reversed(rows):
        row.pop("segment", None)
        print(json.dumps(row, default=str))
    return 0


def cmd_action(args: argparse.Namespace, config: SecurityConfig) -> int:
    params = {"cap": args.cap} if args.cap else {}
    _print(_engine("action", ref=args.ref, action=args.do, params=params))
    return 0


def cmd_benchmark(args: argparse.Namespace, config: SecurityConfig) -> int:
    from . import benchmark

    if args.suite == "latency":
        _print(benchmark.latency(config, rounds=args.rounds))
        return 0
    if args.suite == "replay":
        if not args.data:
            print("--data <file.jsonl> is required for the replay suite", file=sys.stderr)
            return 2
        _print(benchmark.replay(config, Path(args.data)))
        return 0
    print(
        f"The '{args.suite}' suite runs from the repository's benchmarks/ directory with its own dataset "
        f"(see benchmarks/README.md); it is not bundled in the package.",
        file=sys.stderr,
    )
    return 2


def cmd_explainer(args: argparse.Namespace, config: SecurityConfig) -> int:
    print("The Tier 4 explainer (llama.cpp llama-server) is a Stage 2 feature and is not part of this release.")
    return 2


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mcp-shield", description="Semantic LLM firewall for the Model Context Protocol")
    p.add_argument("--version", action="version", version=f"mcp-shield {__version__}")
    p.add_argument("--config", help="path to mcp-shield.toml (default: the per-user config dir)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="check the installation and pick the execution provider")
    s.add_argument("--no-self-test", action="store_true")

    s = sub.add_parser("install", help="route a host's MCP servers through the proxy")
    s.add_argument("--host", required=True, help="claude-desktop | cursor | <path to an MCP config JSON>")
    s.add_argument("--with-rules", action="store_true", help="also insert the host prompt rules")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    s = sub.add_parser("uninstall", help="restore the entries install changed")
    s.add_argument("--host", required=True)
    s.add_argument("--yes", action="store_true")

    s = sub.add_parser("rules", help="print the host prompt rules")
    s.add_argument("--print", action="store_true", dest="do_print", help="print the rules (default)")
    s.add_argument("--short", action="store_true", help="the compact form for tight instruction limits")

    s = sub.add_parser("proxy", help="run a stdio front-end, or open an HTTP listener in the engine")
    s.add_argument("--name", help="server name from the host config (pins and taint key on it)")
    s.add_argument("--listen", help="127.0.0.1:PORT for an HTTP listener")
    s.add_argument("--upstream", help="the Streamable HTTP server URL behind the listener")
    s.add_argument("--bearer-token", help="token the host must present to the listener")

    s = sub.add_parser("engine", help="engine status, or run it")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--foreground", action="store_true", help="run in the foreground (containers, login item)")
    g.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    g.add_argument("--stop", action="store_true")
    s.add_argument(
        "--fail-open", action="store_true", help="DEVELOPMENT ONLY: pass content if the classifier is unavailable"
    )

    s = sub.add_parser("dashboard", help="open the local dashboard")
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--no-browser", action="store_true")

    s = sub.add_parser("pull-model", help="download and verify the classifier weights")
    s.add_argument("--model", default=None)
    s.add_argument("--from", dest="source", help="install a locally built model directory instead")
    s.add_argument("--allow-unpinned", action="store_true", help="development: accept an unpinned Hub revision")
    s.add_argument("--fp16", action="store_true", help="also fetch the fp16 (GPU) graph")

    s = sub.add_parser("pin", help="inspect or accept a server's capability pins")
    s.add_argument("--server", required=True)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true")
    g.add_argument("--accept", action="store_true")
    g.add_argument("--forget", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--yes", action="store_true")

    s = sub.add_parser("events", help="print recent events from the local event store")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--ref", help="filter by reference ID (or tool / server / reason text)")

    s = sub.add_parser("action", help="apply a dashboard action to an event from the terminal")
    s.add_argument("--ref", required=True)
    s.add_argument(
        "--do",
        required=True,
        choices=[
            "allow_content",
            "allow_once",
            "allow_flow",
            "raise_cap",
            "mark_benign",
            "mark_attack",
            "report_fp",
            "accept_pins",
        ],
    )
    s.add_argument("--cap", type=int)

    s = sub.add_parser("benchmark", help="run an evaluation suite")
    s.add_argument("--suite", default="latency", choices=["latency", "replay", "agentdojo", "mcpsecbench"])
    s.add_argument("--data")
    s.add_argument("--rounds", type=int, default=20)

    sub.add_parser("explainer", help="Stage 2: Tier 4 explainer")
    return p


HANDLERS = {
    "doctor": cmd_doctor,
    "install": cmd_install,
    "uninstall": cmd_uninstall,
    "rules": cmd_rules,
    "engine": cmd_engine,
    "dashboard": cmd_dashboard,
    "pull-model": cmd_pull_model,
    "pin": cmd_pin,
    "events": cmd_events,
    "action": cmd_action,
    "benchmark": cmd_benchmark,
    "explainer": cmd_explainer,
}


def utf8_when_redirected() -> None:
    """Windows encodes redirected output in the ANSI code page (cp1252): the rules' dashes and
    symbols would come out garbled or raise. Consoles already get Unicode; pipes and files get UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        isatty = getattr(stream, "isatty", None)
        if reconfigure is not None and isatty is not None and not isatty():
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_when_redirected()
    argv = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1 :]
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config) if args.config else None)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if args.command == "proxy":
        return cmd_proxy(args, config, command)
    if command:
        print("unexpected arguments after --", file=sys.stderr)
        return 2
    try:
        return HANDLERS[args.command](args, config)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
