"""Kalıcı indeks: parçalar (JSONL) + yoğun vektörler (``.npy``) + meta veri.

BM25 istatistikleri diskte tutulmaz; yüklemede yeniden hesaplanır (binlerce parça için
milisaniyeler sürer) ve böylece analizör değişiklikleri yeniden indeksleme gerektirmez.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from turkce_rag.analysis import DEFAULT_ANALYZER, Analyzer
from turkce_rag.bm25 import BM25
from turkce_rag.chunking import ChunkConfig, chunk_documents
from turkce_rag.embeddings import Embedder
from turkce_rag.models import Chunk, Document

FORMAT_VERSION = 1


class IndexMismatchError(RuntimeError):
    pass


class Index:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        embeddings: np.ndarray | None = None,
        embedder_name: str | None = None,
        analyzer: Analyzer = DEFAULT_ANALYZER,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks or [])
        self.embeddings = embeddings
        self.embedder_name = embedder_name
        self.analyzer = analyzer
        if embeddings is not None and len(embeddings) != len(self.chunks):
            raise ValueError("embeddings ve chunks sayısı uyuşmuyor")
        self._bm25: BM25 | None = None
        self._terms: list[set[str]] | None = None

    # ------------------------------------------------------------------ build
    def add_documents(
        self,
        docs: Iterable[Document],
        embedder: Embedder | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> list[Chunk]:
        docs = list(docs)
        new_ids = {d.doc_id for d in docs}
        # Aynı belge yeniden eklenirse eski parçaları değiştir (idempotent ingest).
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id not in new_ids]
        self.chunks = [self.chunks[i] for i in keep]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[keep]

        new_chunks = chunk_documents(docs, chunk_config)
        if embedder is not None:
            if self.embedder_name not in (None, embedder.name) and self.chunks:
                raise IndexMismatchError(
                    f"İndeks '{self.embedder_name}' ile oluşturulmuş, '{embedder.name}' verildi"
                )
            vecs = embedder.embed_documents([c.indexed_text() for c in new_chunks])
            if self.embeddings is None or not len(self.chunks):
                self.embeddings = vecs
            else:
                self.embeddings = np.vstack([self.embeddings, vecs])
            self.embedder_name = embedder.name
        elif self.embeddings is not None and new_chunks:
            raise IndexMismatchError("Bu indeks yoğun vektör içeriyor; embedder vermelisiniz")
        self.chunks.extend(new_chunks)
        self._bm25 = None
        self._terms = None
        return new_chunks

    # ----------------------------------------------------------------- lookup
    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.indexed_text() for c in self.chunks], self.analyzer)
        return self._bm25

    @property
    def chunk_terms(self) -> list[set[str]]:
        if self._terms is None:
            self._terms = [set(t) for t in self.bm25.doc_terms]
        return self._terms

    @property
    def has_dense(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    def __len__(self) -> int:
        return len(self.chunks)

    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        emb_path = p / "embeddings.npy"
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings.astype(np.float32))
        elif emb_path.exists():
            emb_path.unlink()
        meta = {
            "format_version": FORMAT_VERSION,
            "embedder": self.embedder_name,
            "n_chunks": len(self.chunks),
            "documents": self.doc_ids(),
        }
        (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    @classmethod
    def load(cls, path: str | Path, analyzer: Analyzer = DEFAULT_ANALYZER) -> Index:
        p = Path(path)
        meta_path = p / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"İndeks bulunamadı: {p} (önce 'turkce-rag ingest' çalıştırın)")
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise IndexMismatchError(f"Desteklenmeyen indeks sürümü: {meta.get('format_version')}")
        with (p / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        emb_path = p / "embeddings.npy"
        emb = np.load(emb_path) if emb_path.exists() else None
        return cls(chunks, emb, meta.get("embedder"), analyzer)
