"""Per-tool privilege class (02 §2, Tier 1 item 5).

The class is the *highest* of the developer's config, a name heuristic and the server's
tool annotations. Annotations can only raise a class, never lower it — a malicious
server would otherwise mark ``execute_shell`` read-only to escape taint enforcement.
A tool matched by nothing defaults to ``write`` (fail closed).
"""

from __future__ import annotations

from typing import Any

from ..config import PRIVILEGE_RANK, ToolPolicy
from .argpolicy import snake

READ, WRITE, EGRESS, EXECUTE = "read", "write", "egress", "execute"
HIGH = frozenset({WRITE, EGRESS, EXECUTE})

_EXECUTE_PREFIXES = ("execute_", "exec_", "run_", "shell")
_EXECUTE_NAMES = frozenset({"bash", "sh", "zsh", "exec", "execute", "eval", "shell", "terminal", "powershell", "cmd"})
_EGRESS_PREFIXES = ("send_", "post_", "fetch_", "upload_", "publish_", "email_", "notify_", "http_", "webhook_")
_WRITE_PREFIXES = (
    "write_",
    "delete_",
    "update_",
    "create_",
    "edit_",
    "remove_",
    "move_",
    "rename_",
    "insert_",
    "set_",
    "put_",
    "patch_",
    "append_",
    "modify_",
    "add_",
    "upsert_",
    "drop_",
)
_READ_PREFIXES = ("get_", "list_", "search_", "read_", "query_", "find_", "describe_", "show_", "view_", "lookup_")


def name_class(name: str) -> str | None:
    """The name heuristic; ``None`` when the name matches no rule."""
    n = snake(name)
    if n in _EXECUTE_NAMES or n.startswith(_EXECUTE_PREFIXES):
        return EXECUTE
    if n.startswith(_EGRESS_PREFIXES):
        return EGRESS
    if n.startswith(_WRITE_PREFIXES):
        return WRITE
    if n.startswith(_READ_PREFIXES):
        return READ
    return None


def annotation_class(annotations: Any) -> str | None:
    """``openWorldHint`` → egress, ``destructiveHint`` → at least write; explicit ``true`` only."""
    if not isinstance(annotations, dict):
        return None
    if annotations.get("openWorldHint") is True:
        return EGRESS
    if annotations.get("destructiveHint") is True:
        return WRITE
    return None


def highest(*classes: str | None) -> str | None:
    present = [c for c in classes if c]
    return max(present, key=lambda c: PRIVILEGE_RANK[c]) if present else None


def privilege_of(tool: dict[str, Any] | None, name: str, policy: ToolPolicy) -> str:
    annotations = tool.get("annotations") if isinstance(tool, dict) else None
    return highest(policy.privilege, name_class(name), annotation_class(annotations)) or WRITE
