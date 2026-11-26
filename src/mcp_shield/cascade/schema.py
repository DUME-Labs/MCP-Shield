"""Argument validation against a pinned ``inputSchema`` (JSON Schema 2020-12).

References are resolved only inside the schema itself; a network ``$ref`` is never
fetched — it is reported as a violation instead.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable


def _refuse(uri: str) -> Any:
    raise NoSuchResource(ref=uri)  # type: ignore[call-arg]


_REGISTRY: Registry = Registry(retrieve=_refuse)  # type: ignore[call-arg]


class SchemaUnusable(Exception):
    """The server's schema itself is invalid; validation is skipped and reported."""


@lru_cache(maxsize=1024)
def _validator(schema_json: str) -> Draft202012Validator:
    schema = json.loads(schema_json)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise SchemaUnusable(str(exc.message)) from exc
    return Draft202012Validator(schema, registry=_REGISTRY)


def validate_arguments(schema: dict[str, Any] | None, arguments: Any) -> list[str]:
    """Return human-readable violations (empty when the arguments conform).

    Raises :class:`SchemaUnusable` when the schema itself cannot be used.
    """
    if not isinstance(schema, dict) or not schema:
        return []
    validator = _validator(json.dumps(schema, sort_keys=True))
    try:
        errors = sorted(validator.iter_errors(arguments if arguments is not None else {}), key=lambda e: list(e.path))
        return [f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message[:200]}" for e in errors[:10]]
    except Unresolvable as exc:
        return [f"schema reference cannot be resolved locally: {exc}"]
