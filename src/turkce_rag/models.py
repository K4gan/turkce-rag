"""Çekirdek veri yapıları."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Block:
    """Bir belgenin konumu bilinen metin parçası (PDF sayfası ya da Markdown bölümü)."""

    text: str
    page: int | None = None
    section: str | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    page: int | None = None
    section: str | None = None

    def location(self) -> str:
        """İnsan tarafından okunabilir konum etiketi: ``Belge, s. 3`` / ``Belge › Bölüm``."""
        parts = [self.title]
        if self.page is not None:
            parts.append(f"s. {self.page}")
        if self.section:
            parts.append(self.section)
        return " › ".join(parts) if self.page is None else ", ".join(parts)

    def indexed_text(self) -> str:
        """Aranan metin: başlık ve bölüm yolu parçanın önüne eklenir.

        Parçalar tek başına bağlamını kaybedebilir ("Süre 72 saattir." gibi). Başlığı eklemek
        hem BM25'e hem yoğun modele "bu parça neyle ilgili" bilgisini verir.
        """
        header = " › ".join(p for p in (self.title, self.section) if p)
        return f"{header}\n{self.text}" if header else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)


@dataclass
class Hit:
    """Bir sorgu için bulunan parça ve skorları."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    coverage: float = 0.0  # sorgudaki içerik köklerinin parçada geçme oranı


@dataclass(frozen=True)
class Source:
    """Cevapta atıf yapılan kaynak; ``[n]`` numarası cevap metnindeki etiketle aynıdır."""

    n: int
    doc_id: str
    title: str
    source: str
    location: str
    page: int | None
    section: str | None
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    confident: bool
    sources: list[Source]
    hits: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "confident": self.confident,
            "sources": [asdict(s) for s in self.sources],
        }
