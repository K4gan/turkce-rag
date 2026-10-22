"""turkce-rag: Türkçe öncelikli, kaynak gösteren hibrit RAG motoru."""

from turkce_rag.models import Answer, Chunk, Document, Hit, Source
from turkce_rag.normalize import fold_diacritics, normalize_text, turkish_lower, turkish_upper
from turkce_rag.pipeline import RAGEngine

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Hit",
    "RAGEngine",
    "Source",
    "fold_diacritics",
    "normalize_text",
    "turkish_lower",
    "turkish_upper",
]

__version__ = "0.1.0"
