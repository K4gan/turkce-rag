"""Arama kalitesi ölçümü: recall@k ve MRR.

Bir parça, sorunun ``doc_id``'sine ait olup ``evidence`` metnini içeriyorsa "ilgili" sayılır.
Kanıt metni üzerinden tanımlamak, parçalama ayarları değişse bile altın etiketleri geçerli tutar.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from turkce_rag.models import Chunk


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    doc_id: str | None
    evidence: str | None

    @property
    def answerable(self) -> bool:
        return self.doc_id is not None


def load_qa(path: str | Path) -> list[QAItem]:
    with Path(path).open(encoding="utf-8") as f:
        return [QAItem(**json.loads(line)) for line in f if line.strip()]


def is_relevant(item: QAItem, chunk: Chunk) -> bool:
    if item.doc_id is None or item.evidence is None:
        return False
    return chunk.doc_id == item.doc_id and " ".join(item.evidence.split()) in chunk.text


def first_relevant_rank(item: QAItem, ranked: Sequence[Chunk]) -> int | None:
    for r, c in enumerate(ranked, start=1):
        if is_relevant(item, c):
            return r
    return None


@dataclass
class RetrievalReport:
    name: str
    n: int
    recall: dict[int, float]
    mrr: float
    misses: list[str]


def evaluate_retrieval(
    name: str,
    items: Sequence[QAItem],
    search: Callable[[str, int], Sequence[Chunk]],
    ks: Sequence[int] = (1, 3, 5),
    mrr_k: int = 10,
) -> RetrievalReport:
    answerable = [it for it in items if it.answerable]
    depth = max(max(ks), mrr_k)
    hits_at = dict.fromkeys(ks, 0)
    rr_sum = 0.0
    misses: list[str] = []
    for it in answerable:
        rank = first_relevant_rank(it, search(it.question, depth))
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is not None and rank <= mrr_k:
            rr_sum += 1.0 / rank
        if rank is None or rank > max(ks):
            misses.append(it.id)
    n = len(answerable)
    return RetrievalReport(
        name=name,
        n=n,
        recall={k: hits_at[k] / n for k in ks},
        mrr=rr_sum / n,
        misses=misses,
    )
