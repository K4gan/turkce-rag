"""Örnek derlemdeki kurgusal PDF'i (çalışan el kitabı) yeniden üretir.

Kullanım: python scripts/make_sample_pdf.py  (reportlab ve DejaVuSans gerektirir)
Her bölüm ayrı bir sayfaya yazılır; böylece sayfa numaralı atıflar test edilebilir.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
]

TITLE = "Mavi Liman Yazılım Çalışan El Kitabı"
NOTE = "Bu belge turkce-rag örnek derlemi için yazılmış kurgusal bir şirket el kitabıdır."

PAGES: list[tuple[str, list[str]]] = [
    (
        "Çalışma Düzeni ve Uzaktan Çalışma",
        [
            "Mavi Liman Yazılım'da haftalık çalışma süresi 40 saattir. Ekipler arası toplantıların "
            "rahatça planlanabilmesi için hafta içi her gün 10.00 ile 16.00 arası çekirdek çalışma "
            "saatleri olarak kabul edilir; bu saatler dışında çalışanlar başlangıç ve bitiş "
            "saatlerini kendileri belirleyebilir.",
            "Çalışanlar haftada en fazla üç gün uzaktan çalışabilir. Salı günleri tüm ekiplerin "
            "ofiste bulunduğu ortak ofis günüdür. Uzaktan çalışma için şirket, yılda bir kez "
            "verilen ev ofisi desteği kapsamında ergonomik sandalye ve ek monitör sağlar.",
        ],
    ),
    (
        "Yıllık İzin",
        [
            "Çalışanlar, İş Kanunu'nda öngörülen yasal yıllık izin haklarına ek olarak her yıl "
            "iki gün ek izin kullanır. Doğum günü izni ayrıca verilir ve doğum gününün bulunduğu "
            "ay içinde kullanılmalıdır.",
            "İzin talepleri, izin başlangıcından en az beş iş günü önce Pusula adlı insan "
            "kaynakları portalı üzerinden yapılır ve ekip lideri tarafından onaylanır. "
            "Kullanılmayan ek izin günleri bir sonraki yıla devretmez.",
        ],
    ),
    (
        "Eğitim ve Gelişim",
        [
            "Her çalışan için yıllık 12.000 TL tutarında kişisel eğitim bütçesi ayrılır. Bu bütçe "
            "çevrim içi kurslar, kitaplar, sertifika sınavları ve konferans katılımları için "
            "kullanılabilir.",
            "Çalışanlar ayda bir cuma öğleden sonrasını, iş dışı bir öğrenme projesine ayırabilir; "
            "şirket içinde bu uygulamaya Keşif Cuması denir. Konferansta konuşmacı olarak kabul "
            "edilen çalışanların yol ve konaklama giderleri eğitim bütçesinden düşülmez, "
            "şirket tarafından ayrıca karşılanır.",
        ],
    ),
]


def main(out: Path) -> None:
    font = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    if font is None:
        sys.exit("DejaVuSans.ttf bulunamadı (Türkçe karakterler için gerekli)")
    pdfmetrics.registerFont(TTFont("DejaVu", font))
    h = ParagraphStyle("h", fontName="DejaVu", fontSize=15, leading=19, spaceAfter=10)
    p = ParagraphStyle("p", fontName="DejaVu", fontSize=11, leading=15, spaceAfter=8)
    small = ParagraphStyle("s", fontName="DejaVu", fontSize=8, leading=10)

    story: list = []
    for i, (heading, paras) in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(heading, h))
        story.extend(Paragraph(t, p) for t in paras)
        story.append(Spacer(1, 12))
        story.append(Paragraph(NOTE, small))

    doc = SimpleDocTemplate(str(out), pagesize=A4, title=TITLE, author="turkce-rag")
    doc.build(story)
    print(f"yazıldı: {out}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "data" / "sample" / "mavi_liman_el_kitabi.pdf")
