"""Reciprocal Rank Fusion (Cormack vd., 2009).

BM25 skorları sınırsız, kosinüs benzerlikleri [-1, 1] aralığındadır; doğrudan toplanamazlar.
RRF yalnızca sıraları kullanır: ``score(d) = Σ w_i / (k + rank_i(d))``. Kalibrasyon gerektirmez
ve küçük derlemlerde sağlamdır.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Sıralı listeleri birleştirir. Her liste en iyiden en kötüye ID içerir (rank 1'den başlar).

    Eşitlikte ilk listedeki daha iyi sıra, sonra ID'nin ilk görülme sırası kazanır (deterministik).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights ve rankings uzunlukları eşit olmalı")
    scores: dict[K, float] = {}
    first_seen: dict[K, int] = {}
    order = 0
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            if key not in first_seen:
                first_seen[key] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
