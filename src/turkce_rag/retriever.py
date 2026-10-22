"""Hibrit arama: BM25 + yoğun vektör, Reciprocal Rank Fusion ile birleştirilir."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from turkce_rag.analysis import coverage
from turkce_rag.embeddings import Embedder
from turkce_rag.fusion import reciprocal_rank_fusion
from turkce_rag.index import Index, IndexMismatchError
from turkce_rag.models import Hit

Mode = Literal["bm25", "dense", "hybrid"]


class Reranker(Protocol):
    """İsteğe bağlı ikinci aşama sıralayıcı (ör. cross-encoder)."""

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder sarmalayıcısı.

    Örnek: ``CrossEncoderReranker("BAAI/bge-reranker-v2-m3")`` (çok dilli, Türkçe destekli).
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, scores, strict=True):
            h.score = float(s)
        return sorted(hits, key=lambda h: -h.score)


class HybridRetriever:
    def __init__(
        self,
        index: Index,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_pool: int = 50,
        weights: tuple[float, float] = (1.0, 1.0),
        rerank_depth: int = 20,
    ) -> None:
        if index.has_dense and embedder is not None and index.embedder_name != embedder.name:
            raise IndexMismatchError(
                f"İndeks '{index.embedder_name}' ile oluşturulmuş, "
                f"sorgu embedder'ı '{embedder.name}'"
            )
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.weights = weights
        self.rerank_depth = rerank_depth

    @property
    def default_mode(self) -> Mode:
        return "hybrid" if self._dense_available else "bm25"

    @property
    def _dense_available(self) -> bool:
        return self.embedder is not None and self.index.has_dense

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        return self.index.bm25.search(query, self.candidate_pool)

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        assert self.embedder is not None and self.index.embeddings is not None
        q = self.embedder.embed_query(query)
        sims = self.index.embeddings @ q
        k = min(self.candidate_pool, len(sims))
        top = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in top]

    def search(self, query: str, k: int = 5, mode: Mode | None = None) -> list[Hit]:
        if not len(self.index):
            return []
        mode = mode or self.default_mode
        if mode in ("dense", "hybrid") and not self._dense_available:
            raise ValueError(f"'{mode}' modu için yoğun vektörlü indeks ve embedder gerekli")

        bm = self._bm25_ranking(query) if mode in ("bm25", "hybrid") else []
        de = self._dense_ranking(query) if mode in ("dense", "hybrid") else []
        bm_rank = {i: r for r, (i, _) in enumerate(bm, start=1)}
        de_rank = {i: r for r, (i, _) in enumerate(de, start=1)}
        de_score = dict(de)

        if mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = de
        else:
            fused = reciprocal_rank_fusion(
                [[i for i, _ in bm], [i for i, _ in de]], k=self.rrf_k, weights=self.weights
            )

        q_terms = self.index.analyzer(query)
        bm25 = self.index.bm25
        hits: list[Hit] = []
        depth = k if self.reranker is None else max(k, self.rerank_depth)
        for i, score in fused[:depth]:
            hits.append(
                Hit(
                    chunk=self.index.chunks[i],
                    score=float(score),
                    bm25_rank=bm_rank.get(i),
                    dense_rank=de_rank.get(i),
                    dense_score=de_score.get(i),
                    coverage=coverage(q_terms, self.index.chunk_terms[i], bm25.idf, bm25.max_idf),
                )
            )
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]
