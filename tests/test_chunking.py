import itertools

from turkce_rag.chunking import ChunkConfig, chunk_document, split_sentences
from turkce_rag.models import Block, Document


def test_abbreviations_do_not_split() -> None:
    text = "Prof. Dr. Ayşe Yılmaz konuştu. Toplantı Atatürk Cad. No. 5 adresinde yapıldı."
    assert split_sentences(text) == [
        "Prof. Dr. Ayşe Yılmaz konuştu.",
        "Toplantı Atatürk Cad. No. 5 adresinde yapıldı.",
    ]


def test_ordinals_and_numbers() -> None:
    text = "19. yüzyılda kuruldu. 1. Dünya Savaşı 1914'te başladı. Oran 3,5 oldu. Tarih 29.10.1923."
    assert split_sentences(text) == [
        "19. yüzyılda kuruldu.",
        "1. Dünya Savaşı 1914'te başladı.",
        "Oran 3,5 oldu.",
        "Tarih 29.10.1923.",
    ]


def test_question_exclamation_ellipsis_and_quotes() -> None:
    text = 'Gelecek mi? Evet! Bekledik… "Geldi" dedi.'
    assert split_sentences(text) == ["Gelecek mi?", "Evet!", "Bekledik…", '"Geldi" dedi.']


def test_lowercase_after_period_is_not_a_boundary() -> None:
    assert split_sentences("Kitap, defter vb. malzemeler alındı.") == [
        "Kitap, defter vb. malzemeler alındı."
    ]


def test_paragraphs_always_split() -> None:
    assert split_sentences("Birinci paragraf\n\nİkinci paragraf") == [
        "Birinci paragraf",
        "İkinci paragraf",
    ]


def _doc(blocks: list[Block]) -> Document:
    return Document(doc_id="d", title="Belge", source="mem", blocks=blocks)


def test_chunks_respect_size_and_overlap() -> None:
    sents = [f"Bu {i}. cümle biraz uzunca bir Türkçe cümledir." for i in range(1, 30)]
    doc = _doc([Block(text=" ".join(sents), section="Bölüm")])
    chunks = chunk_document(doc, ChunkConfig(max_chars=200, overlap_sentences=1))
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.text) <= 200 + 60  # tek bir cümle sınırı aşabilir ama çok değil
        assert c.section == "Bölüm"
    # Ardışık parçalar bir cümle paylaşır
    for a, b in itertools.pairwise(chunks):
        assert split_sentences(a.text)[-1] == split_sentences(b.text)[0]


def test_chunks_never_cross_page_or_section() -> None:
    doc = _doc(
        [
            Block(text="Birinci sayfa metni. Kısa.", page=1),
            Block(text="İkinci sayfa metni. Bu da kısa.", page=2),
        ]
    )
    chunks = chunk_document(doc)
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].location() == "Belge, s. 1"


def test_chunk_ids_are_deterministic() -> None:
    doc = _doc([Block(text="Aynı metin. Aynı kimlik.")])
    assert chunk_document(doc)[0].chunk_id == chunk_document(doc)[0].chunk_id


def test_short_tail_is_merged() -> None:
    sents = ["A" * 90 + ".", "B" * 90 + ".", "Son."]
    doc = _doc([Block(text=" ".join(sents))])
    chunks = chunk_document(doc, ChunkConfig(max_chars=150, overlap_sentences=0, min_chars=40))
    assert chunks[-1].text.endswith("Son.")
    assert all(len(c.text) > 10 for c in chunks)


def test_paragraph_breaks_survive_packing() -> None:
    text = "Yıllık İzin\n\nÇalışanlar iki gün ek izin kullanır. Ayrıca doğum günü izni var."
    (chunk,) = chunk_document(_doc([Block(text=text)]))
    assert chunk.text == text
    assert split_sentences(chunk.text)[0] == "Yıllık İzin"
