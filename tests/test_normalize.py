import unicodedata

from turkce_rag.normalize import fold_diacritics, normalize_text, turkish_lower, turkish_upper


def test_turkish_lower_dotted_and_dotless_i() -> None:
    assert turkish_lower("IĞDIR") == "ığdır"
    assert turkish_lower("İZMİR") == "izmir"
    # Python'un varsayılanı "i̇" (i + U+0307) üretir; biz üretmemeliyiz.
    assert "\u0307" not in turkish_lower("İstanbul")
    assert "İstanbul".lower() != turkish_lower("İstanbul")


def test_turkish_upper() -> None:
    assert turkish_upper("istanbul") == "İSTANBUL"
    assert turkish_upper("ılık") == "ILIK"


def test_decomposed_input_is_composed_first() -> None:
    decomposed = unicodedata.normalize("NFD", "İşçi Şükrü")
    assert turkish_lower(decomposed) == "işçi şükrü"


def test_fold_diacritics() -> None:
    assert fold_diacritics("Çığ düştü, şöyle böyle") == "Cig dustu, soyle boyle"
    assert fold_diacritics("şikâyet") == "sikayet"
    assert fold_diacritics("İĞNE") == "IGNE"


def test_normalize_text_cleans_pdf_artifacts() -> None:
    raw = "bilgi-\nsayar ve ﬁzik\u00ad dersi  İstanbul’da\r\n\n\n\nyeni\u200b paragraf"
    out = normalize_text(raw)
    assert "bilgisayar" in out
    assert "fizik dersi" in out
    assert "İstanbul'da" in out
    assert "\n\nyeni paragraf" in out
    assert "\u200b" not in out


def test_normalize_keeps_hyphen_before_capital() -> None:
    assert "Ankara-\nİstanbul" in normalize_text("Ankara-\nİstanbul hattı")
