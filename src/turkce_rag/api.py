"""FastAPI tabanlı küçük HTTP API (``pip install 'turkce-rag[api]'``).

Uç noktalar: ``GET /health``, ``POST /search``, ``POST /ask``, ``POST /ingest``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from turkce_rag import __version__
from turkce_rag.embeddings import make_embedder
from turkce_rag.index import Index
from turkce_rag.llm import make_llm
from turkce_rag.loaders import iter_sources
from turkce_rag.pipeline import RAGEngine

ModeT = Literal["bm25", "dense", "hybrid"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    mode: ModeT | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: ModeT | None = None


class IngestRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=20)


def create_app(
    index_dir: str | Path | None = None,
    embedder: str | None = "e5",
    llm: str | None = None,
    engine: RAGEngine | None = None,
) -> FastAPI:
    """Uygulamayı kurar. Testlerde hazır bir ``engine`` verilebilir."""
    if engine is None:
        try:
            index = Index.load(index_dir) if index_dir else Index()
        except FileNotFoundError:
            index = Index()
        emb = make_embedder(embedder) if (index.has_dense or not len(index)) else None
        engine = RAGEngine(index=index, embedder=emb, llm=make_llm(llm))
    lock = threading.Lock()
    app = FastAPI(title="turkce-rag", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "chunks": len(engine.index),
            "documents": len(engine.index.doc_ids()),
            "embedder": engine.index.embedder_name,
            "llm": engine.llm.name,
        }

    @app.post("/search")
    def search(req: SearchRequest) -> list[dict[str, Any]]:
        try:
            hits = engine.search(req.query, k=req.top_k, mode=req.mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return [
            {
                "score": h.score,
                "location": h.chunk.location(),
                "chunk": h.chunk.to_dict(),
                "bm25_rank": h.bm25_rank,
                "dense_rank": h.dense_rank,
            }
            for h in hits
        ]

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        try:
            return engine.ask(req.question, k=req.top_k, mode=req.mode).to_dict()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        bad = [u for u in req.urls if not u.startswith(("http://", "https://"))]
        if bad:
            # API üzerinden yerel dosya sistemi okunmasına izin verme.
            raise HTTPException(status_code=400, detail=f"Yalnızca http(s) URL kabul edilir: {bad}")
        with lock:
            added = engine.add_documents(iter_sources(req.urls))
            if index_dir:
                engine.save(index_dir)
        return {"added_chunks": added, "total_chunks": len(engine.index)}

    return app
