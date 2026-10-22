from pathlib import Path

import pytest

from turkce_rag.loaders import html_to_document, iter_sources, load_markdown, load_pdf, slugify


def test_slugify_turkish() -> None:
    assert slugify("Türk Kahvesi ve Çay") == "turk-kahvesi-ve-cay"
    assert slugify("İŞ KANUNU") == "is-kanunu"


def test_markdown_sections(tmp_path: Path) -> None:
    md = tmp_path / "ornek.md"
    md.write_text(
        "# Başlık\n\nGiriş metni.\n\n## Bölüm A\n\nA metni.\n\n### Alt\n\nAlt metni.\n\n"
        "```\n# kod içindeki başlık değil\n```\n\n## Bölüm B\n\nB metni.\n",
        encoding="utf-8",
    )
    doc = load_markdown(md)
    assert doc.title == "Başlık"
    assert [(b.section, b.text) for b in doc.blocks] == [
        (None, "Giriş metni."),
        ("Bölüm A", "A metni."),
        # Kod bloğundaki "#" başlık sayılmaz; metin olarak bölümde kalır
        ("Bölüm A › Alt", "Alt metni.\n\n# kod içindeki başlık değil"),
        ("Bölüm B", "B metni."),
    ]


def test_pdf_pages(sample_dir: Path) -> None:
    doc = load_pdf(sample_dir / "mavi_liman_el_kitabi.pdf")
    assert doc.title == "Mavi Liman Yazılım Çalışan El Kitabı"
    assert [b.page for b in doc.blocks] == [1, 2, 3]
    assert "Pusula" in doc.blocks[1].text
    assert "ı" in doc.blocks[0].text  # Türkçe karakterler korunur
    # Başlık satırı ilk cümleye yapışmaz
    assert doc.blocks[1].text.startswith("Yıllık İzin\n\nÇalışanlar")


def test_pdf_heading_heuristic_keeps_wrapped_sentences() -> None:
    from turkce_rag.loaders import _separate_pdf_headings

    text = "Başlık Burada\nİlk cümle uzun bir satır olup\nAnkara ile devam eder."
    assert _separate_pdf_headings(text) == (
        "Başlık Burada\n\nİlk cümle uzun bir satır olup\nAnkara ile devam eder."
    )


def test_html_to_document_strips_chrome() -> None:
    html = """<html><head><title>Örnek Sayfa</title><script>x()</script></head><body>
    <nav>Menü</nav><article><h1>Ana</h1><p>Giriş.</p><h2>Ayrıntı</h2><p>Detay metni.</p>
    </article><footer>Telif</footer></body></html>"""
    doc = html_to_document(html, "https://ornek.com/yazi")
    assert doc.title == "Örnek Sayfa"
    assert doc.source == "https://ornek.com/yazi"
    texts = " ".join(b.text for b in doc.blocks)
    assert "Menü" not in texts and "Telif" not in texts and "x()" not in texts
    assert any(b.section == "Ayrıntı" and "Detay" in b.text for b in doc.blocks)


def test_iter_sources_directory(sample_dir: Path) -> None:
    docs = list(iter_sources([str(sample_dir)]))
    ids = {d.doc_id for d in docs}
    assert {"turk-kahvesi", "mavi-liman-el-kitabi", "deprem-hazirlik"} <= ids


def test_iter_sources_missing() -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_sources(["/yok/boyle/bir/dosya.md"]))
