"""Host Prompt Rules (H-1 …): the instruction layer MCP-Shield gives the host (02 §1)."""

from __future__ import annotations

import re
from importlib import resources

_RULE = re.compile(r"^\*\*(H-\d+)\s*·")


def full_text() -> str:
    return resources.files("mcp_shield").joinpath("host_rules.md").read_text(encoding="utf-8")


def rules_text() -> str:
    """Parts A–E only: the text a host should actually be given (no preamble or changelog)."""
    text = full_text()
    start = text.find("## Part A")
    end = text.find("## Suggested short form")
    body = text[start:end] if start >= 0 and end > start else text
    return "# MCP-Shield host rules\n\n" + re.sub(r"\[\[([^\]]+)\]\]", r"\1", body).strip().rstrip("-").strip() + "\n"


def short_text() -> str:
    text = full_text()
    section = text.split("## Suggested short form", 1)[1].split("\n---", 1)[0]
    lines = [ln[2:] if ln.startswith("> ") else ln for ln in section.splitlines()[1:]]
    return "\n".join(ln for ln in lines if ln.strip()).strip() + "\n"


def rule_ids() -> list[str]:
    return [m.group(1) for ln in full_text().splitlines() if (m := _RULE.match(ln))]


def version() -> str:
    versions = re.findall(r"^\|(v\d+\.\d+)\|", full_text(), flags=re.M)
    return versions[-1] if versions else "v0"
