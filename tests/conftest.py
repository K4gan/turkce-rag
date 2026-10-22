from __future__ import annotations

from pathlib import Path

import pytest

from turkce_rag.embeddings import HashingEmbedder
from turkce_rag.loaders import iter_sources
from turkce_rag.models import Document
from turkce_rag.pipeline import RAGEngine

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def sample_docs() -> list[Document]:
    return list(iter_sources([str(SAMPLE_DIR)]))


@pytest.fixture()
def engine(sample_docs: list[Document]) -> RAGEngine:
    """Ağ ve model indirmesi gerektirmeyen tam motor: hashing embedder + stub LLM."""
    eng = RAGEngine(embedder=HashingEmbedder())
    eng.add_documents(sample_docs)
    return eng
