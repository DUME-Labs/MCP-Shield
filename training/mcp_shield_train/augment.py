"""Augmentation — on the TRAIN split only (03 §5 step 3).

Tool names, parameter keys and the context prefix are varied so the classifier learns
the instruction, not the surface form (``read_file`` → ``fetch_doc_v2``, ``file_path`` →
``doc_uri``). Attacks can additionally be paraphrased through a callable (the synthetic
generator's ``llama-server``). Validation and test stay untouched.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from mcp_shield.textnorm import context_prefix

from .data import Record

TOOL_SYNONYMS = {
    "read_file": ["fetch_doc_v2", "load_text", "open_document", "get_file_contents"],
    "get_issue": ["fetch_ticket", "issue_lookup", "read_issue_v3"],
    "fetch": ["http_get", "retrieve_url", "download_page"],
    "search": ["web_search", "lookup", "find_pages"],
    "send_email": ["post_message", "mail_send", "notify_user"],
    "run_shell": ["exec_command", "terminal_run", "shell_exec"],
}
KEY_SYNONYMS = {
    "file_path": ["doc_uri", "path", "target"],
    "query": ["q", "search_terms"],
    "url": ["href", "link"],
    "command": ["cmd", "script"],
    "body": ["content", "message_text"],
}
FIELDS = [
    "content[0].text",
    "content[1].text",
    "structuredContent.body",
    "contents[0].text",
    "messages[0].content.text",
    "description",
    "inputSchema.properties.query.description",
]
SERVERS = ["github", "filesystem", "web", "notes", "jira", "slack", "docs", "postgres", "mail", "wiki"]
_PREFIX = re.compile(r"^\[server=(?P<server>\S+) tool=(?P<tool>\S+) field=(?P<field>\S+)\]$")


def vary_prefix(context: str, rng: random.Random) -> str:
    m = _PREFIX.match(context)
    if not m:
        return context
    tool = m.group("tool")
    tool = rng.choice(TOOL_SYNONYMS.get(tool, [tool])) if rng.random() < 0.7 else tool
    if rng.random() < 0.3:
        tool = f"{tool}_v{rng.randint(2, 9)}"
    server = rng.choice(SERVERS) if rng.random() < 0.5 else m.group("server")
    field = m.group("field")
    if field.startswith(("content", "structuredContent", "contents")) and rng.random() < 0.4:
        field = rng.choice(FIELDS[:5])
    return context_prefix(server, tool, field)


def vary_keys(text: str, rng: random.Random) -> str:
    for key, options in KEY_SYNONYMS.items():
        if key in text and rng.random() < 0.5:
            text = re.sub(rf"\b{re.escape(key)}\b", rng.choice(options), text)
    return text


def augment(
    records: list[Record], copies: int = 1, seed: int = 99, paraphrase: Callable[[Record], str | None] | None = None
) -> list[Record]:
    """Return new augmented records for the train split (originals are not modified)."""
    rng = random.Random(seed)
    out: list[Record] = []
    for r in records:
        if r.split != "train":
            raise ValueError(f"augmentation is train-only; {r.id} is in {r.split!r}")
        if r.extra.get("augmented_from"):
            continue
        for n in range(copies):
            text = vary_keys(r.text, rng)
            if paraphrase is not None and r.label != "benign":
                text = paraphrase(r) or text
            out.append(
                Record(
                    id=f"{r.id}#aug{n}",
                    context=vary_prefix(r.context, rng),
                    text=text,
                    label=r.label,
                    source=r.source,
                    group_id=r.group_id,
                    split="train",
                    hard_negative=r.hard_negative,
                    extra={**r.extra, "augmented_from": r.id},
                )
            )
    return out
