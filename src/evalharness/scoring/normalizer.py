"""Text normalization for lexical metrics."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from evalharness.hashing import sha256_canonical

ARTICLES = {"a", "an", "the"}


@dataclass(frozen=True)
class NormalizerConfig:
    lowercase: bool = True
    strip_articles: bool = True
    strip_punctuation: bool = True
    collapse_whitespace: bool = True
    unicode_nfkc: bool = True
    numeric_tol: float | None = None
    version: str = "1.0.0"


class Normalizer:
    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()

    @property
    def config_id(self) -> str:
        return sha256_canonical(self.config.__dict__)

    def normalize(self, text: str) -> str:
        value = text
        if self.config.unicode_nfkc:
            value = unicodedata.normalize("NFKC", value)
        if self.config.lowercase:
            value = value.lower()
        if self.config.strip_punctuation:
            value = re.sub(r"[^\w\s.-]", " ", value)
        if self.config.numeric_tol is not None:
            value = self._canonicalize_numbers(value)
        if self.config.strip_articles:
            value = " ".join(w for w in value.split() if w not in ARTICLES)
        if self.config.collapse_whitespace:
            value = re.sub(r"\s+", " ", value).strip()
        return value

    def _canonicalize_numbers(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            num = float(match.group(0))
            tol = self.config.numeric_tol or 0.0
            rounded = round(num / tol) * tol if tol else num
            if rounded == int(rounded):
                return str(int(rounded))
            return str(rounded)

        return re.sub(r"-?\d+\.?\d*", repl, text)
