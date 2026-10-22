from collections.abc import Sequence

from turkce_rag.llm import NOT_FOUND, Context, parse_citations
from turkce_rag.models import Hit
from turkce_rag.pipeline import RAGEngine


def test_answer_cites_correct_pdf_page(engine: RAGEngine) -> None:
    ans = engine.ask("İzin talebi kaç iş günü önce yapılmalı?")
    assert ans.confident
    assert "beş iş günü" in ans.text
    assert ans.sources, ans
    top = ans.sources[0]
    assert top.doc_id == "mavi-liman-el-kitabi"
    assert top.page == 2
    assert top.location == "Mavi Liman Yazılım Çalışan El Kitabı, s. 2"
    assert f"[{top.n}]" in ans.text


def test_answer_cites_markdown_section(engine: RAGEngine) -> None:
    ans = engine.ask("Peribacaları nasıl oluştu?")
    assert ans.confident
    assert ans.sources[0].doc_id == "kapadokya"
    assert ans.sources[0].section == "Peribacaları"


def test_ascii_question_still_answered(engine: RAGEngine) -> None:
    ans = engine.ask("deprem cantasinda neler olmali")
    assert ans.confident
    assert ans.sources[0].doc_id == "deprem-hazirlik"


def test_unanswerable_question_says_not_found(engine: RAGEngine) -> None:
    for q in ("Kuantum bilgisayarlar nasıl çalışır?", "Espresso nasıl hazırlanır?"):
        ans = engine.ask(q)
        assert not ans.confident, q
        assert ans.text == NOT_FOUND
        assert ans.sources == []


def test_hybrid_bm25_dense_modes(engine: RAGEngine) -> None:
    q = "Karbon tutma işlemini hangi enzim yapar?"
    for mode in ("bm25", "dense", "hybrid"):
        hits = engine.search(q, k=3, mode=mode)
        assert hits[0].chunk.doc_id == "fotosentez", mode
    hybrid = engine.search(q, k=3, mode="hybrid")
    assert hybrid[0].bm25_rank is not None and hybrid[0].dense_rank is not None


class _FixedLLM:
    name = "fixed"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        return self.reply


def test_invalid_citations_are_dropped(engine: RAGEngine) -> None:
    engine.llm = _FixedLLM("Kahve Yemen'den geldi [1][9][1].")
    ans = engine.ask("Türk kahvesi nereden geldi?")
    assert [s.n for s in ans.sources] == [1]


def test_llm_refusal_is_respected(engine: RAGEngine) -> None:
    engine.llm = _FixedLLM(NOT_FOUND)
    ans = engine.ask("Türk kahvesi nereden geldi?")
    assert not ans.confident and ans.sources == []


def test_parse_citations() -> None:
    assert parse_citations("a [2] b [1][2] c [7]", valid=[1, 2, 3]) == [2, 1]
    assert parse_citations("atıf yok", valid=[1]) == []


class _Reverse:
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return list(reversed(hits))


def test_reranker_hook(engine: RAGEngine) -> None:
    q = "Fotosentezde oksijen nereden gelir?"
    base = engine.search(q, k=3)
    engine.reranker = _Reverse()
    reranked = engine.search(q, k=3)
    # Reranker daha derin bir aday havuzunu görür (rerank_depth), sonra k'ya kesilir
    assert len(reranked) == 3
    assert reranked[0].chunk != base[0].chunk


def test_empty_engine() -> None:
    ans = RAGEngine().ask("Herhangi bir soru?")
    assert not ans.confident
