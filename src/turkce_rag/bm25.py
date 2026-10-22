"""Okapi BM25 — bağımlılıksız, şeffaf bir uygulama."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence


class BM25:
    def __init__(
        self,
        docs: Sequence[str],
        analyzer: Callable[[str], list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.analyzer = analyzer
        self.k1 = k1
        self.b = b
        self.doc_terms: list[Counter[str]] = [Counter(analyzer(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.doc_terms]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        df: Counter[str] = Counter()
        for c in self.doc_terms:
            df.update(c.keys())
        n = len(self.doc_terms)
        # Lucene'deki gibi her zaman pozitif kalan IDF.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Derlemde hiç geçmeyen bir terimin alacağı IDF (df = 0).
        self.max_idf = math.log(1 + (n + 0.5) / 0.5) if n else 1.0

    def __len__(self) -> int:
        return len(self.doc_terms)

    def scores(self, query: str) -> list[float]:
        q_terms = self.analyzer(query)
        out = [0.0] * len(self.doc_terms)
        if not q_terms or not self.avgdl:
            return out
        for i, tf in enumerate(self.doc_terms):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
            s = 0.0
            for t in q_terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
