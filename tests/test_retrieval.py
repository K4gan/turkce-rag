import numpy as np
import pytest

from turkce_rag.analysis import Analyzer
from turkce_rag.bm25 import BM25
from turkce_rag.embeddings import HashingEmbedder
from turkce_rag.fusion import reciprocal_rank_fusion

DOCS = [
    "Kitaplarımızdan bazılarını kütüphaneye bağışladık.",
    "Bahçedeki ağaçların yaprakları sonbaharda dökülür.",
    "Deprem çantasında su ve düdük bulunmalıdır.",
]


def test_bm25_turkish_morphology_beats_plain() -> None:
    tr = BM25(DOCS, Analyzer())
    plain = BM25(DOCS, Analyzer.plain())
    assert tr.search("kitap", 3)[0][0] == 0
    assert plain.search("kitap", 3) == []  # düz tokenizasyon çekimli biçimi bulamaz


def test_bm25_ascii_query() -> None:
    tr = BM25(DOCS, Analyzer())
    assert tr.search("agac yapragi", 1)[0][0] == 1
    assert tr.search("dudük", 1)[0][0] == 2


def test_bm25_empty_query_and_corpus() -> None:
    assert BM25(DOCS, Analyzer()).search("ve ile", 3) == []  # yalnızca durak sözcükler
    assert BM25([], Analyzer()).search("kitap", 3) == []


def test_rrf_math_and_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    # a ve b eşit skorlu: önce görülen (a) kazanır → deterministik
    assert [key for key, _ in fused] == ["a", "b", "c", "d"]


def test_rrf_weights() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])
    assert fused[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(dim=64)
    a = e.embed_documents(["Türk kahvesi", "fotosentez"])
    b = e.embed_documents(["Türk kahvesi", "fotosentez"])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert e.embed_documents([]).shape == (0, 64)
    q = e.embed_query("turk kahvesi")
    assert float(a[0] @ q) > float(a[1] @ q)
