"""Uçtan uca RAG motoru: yükle → parçala → indeksle → ara → yanıtla (atıflı)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from turkce_rag.chunking import ChunkConfig
from turkce_rag.embeddings import Embedder, make_embedder
from turkce_rag.index import Index
from turkce_rag.llm import LLM, NOT_FOUND, StubLLM, make_llm, parse_citations
from turkce_rag.loaders import iter_sources
from turkce_rag.models import Answer, Document, Hit, Source
from turkce_rag.retriever import HybridRetriever, Mode, Reranker


@dataclass(frozen=True)
class AnswerPolicy:
    """Ne zaman "bilmiyorum" denileceği.

    Sorgunun içerik köklerinin en az ``min_coverage`` oranı en iyi parçalardan birinde geçmiyorsa
    ve (yoğun arama açıksa) en yüksek kosinüs benzerliği ``min_dense_score`` altındaysa, LLM hiç
    çağrılmadan "bulamadım" yanıtı verilir. Bu, uydurma (hallucination) riskini en ucuz yerde keser.
    """

    min_coverage: float = 0.5
    min_dense_score: float | None = None
    context_k: int = 4


class RAGEngine:
    def __init__(
        self,
        index: Index | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        policy: AnswerPolicy | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        self.index = index or Index()
        self.embedder = embedder
        self.llm = llm or StubLLM()
        self.reranker = reranker
        self.policy = policy or AnswerPolicy()
        self.chunk_config = chunk_config or ChunkConfig()

    # ----------------------------------------------------------- construction
    @classmethod
    def open(
        cls,
        index_dir: str | Path,
        embedder: str | None = "e5",
        llm: str | None = None,
        **kwargs: object,
    ) -> RAGEngine:
        """Diskteki bir indeksi açar. ``embedder`` indeksi oluşturanla aynı olmalıdır."""
        index = Index.load(index_dir)
        emb = make_embedder(embedder) if index.has_dense else None
        return cls(index=index, embedder=emb, llm=make_llm(llm), **kwargs)  # type: ignore[arg-type]

    @property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(self.index, self.embedder, self.reranker)

    # ------------------------------------------------------------------ ingest
    def add_documents(self, docs: Iterable[Document]) -> int:
        return len(self.index.add_documents(docs, self.embedder, self.chunk_config))

    def ingest(self, inputs: Iterable[str]) -> int:
        """Dosya/klasör/URL listesini indeksler; eklenen parça sayısını döndürür."""
        return self.add_documents(iter_sources(inputs))

    def save(self, index_dir: str | Path) -> None:
        self.index.save(index_dir)

    # ------------------------------------------------------------------- query
    def search(self, question: str, k: int = 5, mode: Mode | None = None) -> list[Hit]:
        return self.retriever.search(question, k=k, mode=mode)

    def is_confident(self, hits: list[Hit]) -> bool:
        if not hits:
            return False
        if max(h.coverage for h in hits) >= self.policy.min_coverage:
            return True
        if self.policy.min_dense_score is not None:
            dense = [h.dense_score for h in hits if h.dense_score is not None]
            return bool(dense) and max(dense) >= self.policy.min_dense_score
        return False

    def ask(self, question: str, k: int | None = None, mode: Mode | None = None) -> Answer:
        k = k or self.policy.context_k
        hits = self.search(question, k=k, mode=mode)
        if not self.is_confident(hits):
            return Answer(question=question, text=NOT_FOUND, confident=False, sources=[], hits=hits)

        contexts = [(n, h.chunk) for n, h in enumerate(hits, start=1)]
        text = self.llm.generate(question, contexts).strip()
        if not text or NOT_FOUND.rstrip(".") in text:
            return Answer(question=question, text=NOT_FOUND, confident=False, sources=[], hits=hits)

        cited = parse_citations(text, [n for n, _ in contexts])
        by_n = dict(contexts)
        sources = [
            Source(
                n=n,
                doc_id=by_n[n].doc_id,
                title=by_n[n].title,
                source=by_n[n].source,
                location=by_n[n].location(),
                page=by_n[n].page,
                section=by_n[n].section,
                snippet=by_n[n].text[:240],
            )
            for n in cited
        ]
        return Answer(question=question, text=text, confident=True, sources=sources, hits=hits)
