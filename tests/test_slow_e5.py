"""Gerçek multilingual-e5-small modeliyle duman testi. Yalnızca RUN_SLOW=1 ile çalışır."""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW=1 değil"),
]


def test_e5_hybrid_end_to_end(sample_docs) -> None:  # type: ignore[no-untyped-def]
    from turkce_rag.embeddings import SentenceTransformerEmbedder
    from turkce_rag.pipeline import RAGEngine

    eng = RAGEngine(embedder=SentenceTransformerEmbedder())
    eng.add_documents(sample_docs)
    ans = eng.ask("Bitkiler neden yeşil görünür?")
    assert ans.confident and ans.sources[0].doc_id == "fotosentez"
