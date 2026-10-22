from pathlib import Path

import pytest

from turkce_rag.embeddings import HashingEmbedder
from turkce_rag.index import Index, IndexMismatchError
from turkce_rag.models import Block, Document
from turkce_rag.retriever import HybridRetriever


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, title=doc_id, source=f"{doc_id}.md", blocks=[Block(text=text)])


def test_save_load_roundtrip(tmp_path: Path) -> None:
    emb = HashingEmbedder(dim=32)
    idx = Index()
    idx.add_documents([_doc("a", "Birinci belge."), _doc("b", "İkinci belge.")], emb)
    idx.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert loaded.chunks == idx.chunks
    assert loaded.embedder_name == emb.name
    assert loaded.embeddings is not None and loaded.embeddings.shape == (2, 32)


def test_reingest_replaces_document() -> None:
    emb = HashingEmbedder(dim=32)
    idx = Index()
    idx.add_documents([_doc("a", "Eski metin."), _doc("b", "Başka.")], emb)
    idx.add_documents([_doc("a", "Yeni metin.")], emb)
    assert len(idx) == 2
    assert {c.text for c in idx.chunks} == {"Başka.", "Yeni metin."}
    assert idx.embeddings is not None and len(idx.embeddings) == 2


def test_embedder_mismatch_is_rejected(tmp_path: Path) -> None:
    idx = Index()
    idx.add_documents([_doc("a", "Metin.")], HashingEmbedder(dim=32))
    with pytest.raises(IndexMismatchError):
        idx.add_documents([_doc("b", "Metin.")], HashingEmbedder(dim=64))
    with pytest.raises(IndexMismatchError):
        HybridRetriever(idx, HashingEmbedder(dim=64))


def test_missing_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Index.load(tmp_path / "yok")


def test_bm25_only_index_rejects_dense_mode() -> None:
    idx = Index()
    idx.add_documents([_doc("a", "Metin burada.")])
    r = HybridRetriever(idx)
    assert r.default_mode == "bm25"
    assert r.search("metin", 1)[0].chunk.doc_id == "a"
    with pytest.raises(ValueError):
        r.search("metin", 1, mode="hybrid")
