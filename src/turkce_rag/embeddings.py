"""Yoğun (dense) gömme modelleri.

Varsayılan model ``intfloat/multilingual-e5-small`` (118M parametre, 384 boyut): Türkçe dahil
~100 dil, CPU'da çalışacak kadar küçük. E5 ailesi ``query:`` / ``passage:`` ön ekleriyle
eğitildiği için ön ekler burada otomatik eklenir.

``HashingEmbedder`` model indirmeden çalışan deterministik bir yedektir: karakter n-gram'larını
sabit boyutlu bir vektöre hash'ler. Testler ve CI bunu kullanır; anlamsal değil biçimsel
benzerlik yakalar, ama Türkçe ekleri karakter düzeyinde kısmen eşleştirdiği için işe yarar bir
taban çizgisidir.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from turkce_rag.normalize import fold_diacritics, turkish_lower

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Karakter 3-5 gram hashing; tamamen deterministik, sıfır bağımlılık."""

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        norm = fold_diacritics(turkish_lower(text))
        for word in norm.split():
            w = f"<{word}>"
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(max(1, len(w) - n + 1)):
                    h = hashlib.blake2b(w[i : i + n].encode(), digest_size=8).digest()
                    idx = int.from_bytes(h[:4], "little") % self.dim
                    sign = 1.0 if h[4] & 1 else -1.0
                    v[idx] += sign
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self._vec(text)[None, :])[0]


class SentenceTransformerEmbedder:
    """sentence-transformers sarmalayıcısı (``pip install turkce-rag[dense]``)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - ortama bağlı
            raise ImportError(
                "Yoğun arama için: pip install 'turkce-rag[dense]' (sentence-transformers)"
            ) from e
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        dim = self._model.get_sentence_embedding_dimension()
        self.dim = int(dim) if dim is not None else 0
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefix = "passage: " if self._e5 else ""
        return self._encode([prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        prefix = "query: " if self._e5 else ""
        return self._encode([prefix + text])[0]


def make_embedder(spec: str | None) -> Embedder | None:
    """``"none"`` → yoğun arama kapalı, ``"hash"`` → HashingEmbedder, diğerleri → HF model adı."""
    if spec is None or spec.lower() in {"", "none", "off"}:
        return None
    if spec.lower() in {"hash", "hashing"}:
        return HashingEmbedder()
    if spec.lower() == "e5":
        spec = DEFAULT_MODEL
    return SentenceTransformerEmbedder(spec)
