"""Türkçe'ye duyarlı cümle bölme ve parçalama (chunking).

Naif ``re.split(r"[.!?]")`` Türkçe metinde sürekli yanlış bölünür:

* Kısaltmalar: ``Prof. Dr. Ayşe Yılmaz``, ``vb.``, ``örn.``, ``bkz.``, ``T.C.``, ``Cad.``
* Sıra sayıları: Türkçede sıra sayısı rakamdan sonra nokta ile yazılır: ``19. yüzyıl``,
  ``3. madde``, ``1. Dünya Savaşı`` (büyük harfle devam etse bile cümle sonu değildir).
* Ondalık ve tarih: ``3,5``, ``29.10.1923``.

Parçalar cümle sınırında kesilir, sayfa/bölüm sınırlarını asla aşmaz ve komşu parçalarla bir
cümlelik örtüşme taşır; böylece atıf (belge + sayfa/bölüm) her zaman tek ve doğru bir konumu
gösterir.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from turkce_rag.models import Block, Chunk, Document
from turkce_rag.normalize import turkish_lower

ABBREVIATIONS: frozenset[str] = frozenset(
    [
        "dr",
        "prof",
        "doç",
        "yrd",
        "öğr",
        "gör",
        "av",
        "müh",
        "uzm",
        "arş",
        "sn",
        "bkz",
        "örn",
        "vb",
        "vs",
        "vd",
        "yy",
        "mö",
        "ms",
        "tc",
        "no",
        "nr",
        "s",
        "sf",
        "sy",
        "hz",
        "st",
        "cad",
        "sok",
        "mah",
        "apt",
        "blv",
        "d",
        "tel",
        "faks",
        "fax",
        "alb",
        "yzb",
        "bnb",
        "org",
        "ltd",
        "şti",
        "a.ş",
        "m.ö",
        "m.s",
        "t.c",
        "a.b.d",
        "ab",
        "ç.n",
        "çev",
        "haz",
        "ed",
        "bl",
        "md",
        "mad",
        "fık",
        "krş",
        "ünv",
        "üniv",
        "bşk",
        "gn",
        "müd",
        "yay",
    ]
)

# Aday cümle sonu: noktalama + boşluk(lar) + bir sonraki cümlenin ilk karakteri.
_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*)(\s+)(?=[\"'(\[]?[A-ZÇĞİÖŞÜ0-9])")
_TOKEN_BEFORE = re.compile(r"([\w.]+)$", re.UNICODE)


def _is_abbreviation_or_ordinal(left: str) -> bool:
    m = _TOKEN_BEFORE.search(left)
    if not m:
        return False
    token = m.group(1)
    if token.isdigit():
        # "19. yüzyıl" / "1. Dünya Savaşı": rakam + nokta Türkçede sıra sayısıdır.
        return True
    bare = turkish_lower(token.rstrip("."))
    if bare in ABBREVIATIONS:
        return True
    # Tek büyük harf + nokta: baş harfli adlar ("M. Kemal").
    return len(token) == 1 and token.isupper()


def split_sentences(text: str) -> list[str]:
    """Türkçe metni cümlelere böler. Satır sonları (paragraf) her zaman sınırdır."""
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        start = 0
        for m in _BOUNDARY.finditer(para):
            punct = m.group(1)
            left = para[start : m.start(1)]
            single_period = punct.startswith(".") and len(punct.rstrip("\"')]")) == 1
            if single_period and _is_abbreviation_or_ordinal(left):
                continue
            end = m.end(1)
            sentence = para[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = m.end()
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 700
    overlap_sentences: int = 1
    min_chars: int = 40  # daha kısa kuyruk parçalar bir öncekiyle birleştirilir


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}#{idx}-{digest}"


def _join(window: list[tuple[int, str]]) -> str:
    """Cümleleri birleştirir; paragraf sınırlarını boş satırla korur."""
    out = ""
    prev_para: int | None = None
    for para, sent in window:
        if prev_para is not None:
            out += "\n\n" if para != prev_para else " "
        out += sent
        prev_para = para
    return out


def _pack(sentences: list[tuple[int, str]], cfg: ChunkConfig) -> list[str]:
    windows: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    for item in sentences:
        sent = item[1]
        if current and size + len(sent) + 1 > cfg.max_chars:
            windows.append(current)
            current = current[-cfg.overlap_sentences :] if cfg.overlap_sentences else []
            size = sum(len(s) + 1 for _, s in current)
        current.append(item)
        size += len(sent) + 1
    if current:
        if windows and size < cfg.min_chars:
            # Kısa kuyruk: önceki pencereye ekle (örtüşen cümleleri tekrar etmeden).
            tail = current[cfg.overlap_sentences :] if cfg.overlap_sentences else current
            windows[-1].extend(tail)
        else:
            windows.append(current)
    return [_join(w) for w in windows]


def chunk_block(block: Block, cfg: ChunkConfig) -> list[str]:
    sentences = [
        (p, sent)
        for p, para in enumerate(re.split(r"\n\s*\n", block.text))
        for sent in split_sentences(para)
    ]
    return _pack(sentences, cfg)


def chunk_document(doc: Document, cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []
    for block in doc.blocks:
        for text in chunk_block(block, cfg):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.doc_id, len(chunks), text),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    text=text,
                    page=block.page,
                    section=block.section,
                )
            )
    return chunks


def chunk_documents(docs: Iterable[Document], cfg: ChunkConfig | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, cfg))
    return out
