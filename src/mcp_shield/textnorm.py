"""Text normalisation shared by the runtime extractor and the training pipeline (02 §3).

The classifier must see the text the host LLM effectively sees, so the cheapest
obfuscation routes are closed before classification:

* NFKC folds compatibility look-alikes (full-width letters, ligatures).
* Zero-width and bidi-control characters are removed.
* Unicode *tag* characters (U+E0000–U+E007F) — invisible to people but readable to
  many models, a known way to smuggle instructions — are decoded to the ASCII they
  mirror, so hidden text becomes visible text.
* Long base64 runs are decoded for a second classification pass.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections.abc import Iterator

ZERO_WIDTH = frozenset("​‌‍⁠﻿᠎­͏ᅟᅠㅤﾠ")
BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩‎‏؜")
_STRIP = ZERO_WIDTH | BIDI_CONTROLS
_TAG_START, _TAG_END = 0xE0000, 0xE007F
_B64_RUN = re.compile(r"[A-Za-z0-9+/_-]{40,}={0,2}")
_WS = re.compile(r"\s+")
_LABEL_OK = re.compile(r"[^A-Za-z0-9._:/@\-]")


def normalize(text: str) -> str:
    """NFKC, decode tag characters, drop zero-width and bidi controls."""
    if not text:
        return text
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        cp = ord(ch)
        if _TAG_START <= cp <= _TAG_END:
            inner = cp - _TAG_START
            if 0x20 <= inner < 0x7F:
                out.append(chr(inner))
            continue
        if ch in _STRIP:
            continue
        out.append(ch)
    return "".join(out)


def has_hidden_characters(text: str) -> bool:
    """True when ``text`` contains characters :func:`normalize` removes or decodes."""
    return any(ch in _STRIP or _TAG_START <= ord(ch) <= _TAG_END for ch in text)


def fingerprint_text(text: str) -> str:
    """Canonical form for overlap fingerprints: normalised, case-folded, whitespace collapsed."""
    return _WS.sub(" ", normalize(text).casefold()).strip()


def decoded_base64_runs(text: str, *, limit: int = 8, min_printable: float = 0.9) -> Iterator[str]:
    """Yield readable text hidden in long base64 (or URL-safe base64) runs."""
    found = 0
    for match in _B64_RUN.finditer(text):
        if found >= limit:
            return
        run = match.group(0)
        raw = _try_b64(run)
        if raw is None:
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if decoded and _printable_ratio(decoded) >= min_printable:
            found += 1
            yield decoded


def _try_b64(run: str) -> bytes | None:
    padded = run + "=" * (-len(run) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            return decoder(padded)
        except (binascii.Error, ValueError):
            continue
    return None


def _printable_ratio(text: str) -> float:
    ok = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return ok / len(text)


def sanitize_label(value: object, max_len: int = 64) -> str:
    """Server and tool names are attacker-controlled: restrict them before they reach a prompt."""
    text = normalize(str(value))
    text = _LABEL_OK.sub("_", text)
    return text[:max_len] or "_"


def context_prefix(server: str, tool: str, field: str) -> str:
    """The classifier's input prefix, identical at training and run time (03 §2)."""
    return f"[server={sanitize_label(server)} tool={sanitize_label(tool)} field={sanitize_label(field, 96)}]"
