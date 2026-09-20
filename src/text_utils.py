"""Shared Vietnamese text helpers used by retrieval and offline evaluation."""

from __future__ import annotations

import re
import unicodedata


TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def strip_accents(text: str) -> str:
    """Return lowercase text without Vietnamese accents."""
    normalized = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text consistently for BM25 and fallback scoring."""
    return TOKEN_PATTERN.findall(strip_accents(text))


def token_set(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1}
