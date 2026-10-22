import pytest

from turkce_rag.analysis import Analyzer, coverage, stem


@pytest.mark.parametrize(
    ("forms", "expected_same"),
    [
        (["kitap", "kitabı", "kitaplar", "kitaplarımızdan"], True),
        (["ağaç", "ağacı", "ağaçlar"], True),
        (["renk", "rengi"], True),
        (["deprem", "depremde", "depremden"], True),
        (["veri", "verileri", "verilerin"], True),
        (["hak", "hakkı", "haklar"], True),
    ],
)
def test_inflected_forms_share_a_stem(forms: list[str], expected_same: bool) -> None:
    stems = {stem(f) for f in forms}
    assert (len(stems) == 1) is expected_same, stems


def test_apostrophe_suffix_is_dropped() -> None:
    assert stem("istanbul'da") == "istanbul"
    assert stem("kvkk'ya") == stem("kvkk")


def test_short_words_are_not_overstemmed() -> None:
    assert stem("elma") == "elma"  # tek harfli ek kökü 4 harfin altına indiremez
    assert stem("ev") == "ev"
    assert stem("2016") == "2016"


def test_analyzer_turkish_vs_plain() -> None:
    text = "İstanbul'daki IŞIK ve Çiçekler çok güzeldir."
    assert Analyzer()(text) == ["istanbul", "isik", "cicek", "guzel"]
    plain = Analyzer.plain()(text)
    assert "ve" in plain  # durak sözcük yok
    assert "ışık" not in plain  # varsayılan lower() "IŞIK" → "işik" üretir


def test_ascii_query_matches_turkish_text_only_with_folding() -> None:
    doc = set(Analyzer()("Şikâyet başvurusu otuz gün içinde sonuçlanır."))
    assert set(Analyzer()("sikayet basvurusu")) <= doc
    no_fold = Analyzer(fold=False)
    assert not set(no_fold("sikayet")) & set(no_fold("Şikâyet"))


def test_circumflex_is_orthographic() -> None:
    assert Analyzer(fold=False)("Resmî") == Analyzer(fold=False)("Resmi")


def test_coverage_plain_and_prefix() -> None:
    doc = {"kanun", "yayimlanmistir", "kurul"}
    assert coverage(["kanun"], doc) == 1.0
    # Önek toleransı: hafif kök bulucunun kaçırdığı fiil çekimi
    assert coverage(["yayimlant"], doc) == 1.0
    assert coverage(["xyz"], doc) == 0.0
    assert coverage([], doc) == 0.0


def test_coverage_idf_weighting_penalises_missing_rare_term() -> None:
    doc = {"hazirlanir", "kahv"}
    idf = {"hazirlanir": 1.0}
    # "espresso" derlemde yok → en yüksek IDF ile ağırlıklanır, eksikliği belirleyici olur
    assert coverage(["espresso", "hazirlanir"], doc, idf, default_idf=4.0) == pytest.approx(0.2)
    assert coverage(["espresso", "hazirlanir"], doc) == 0.5


def test_ascii_stopwords_are_removed_too() -> None:
    assert Analyzer()("cunku olmali") == []
    assert Analyzer()("çünkü olmalı") == []
