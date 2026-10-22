"""Takılabilir LLM arka uçları.

* ``stub``    — deterministik, ağsız, çıkarımsal (extractive). Testler/CI ve hızlı denemeler için.
* ``openai``  — OpenAI uyumlu herhangi bir ``/chat/completions`` uç noktası (OpenAI, vLLM,
  LM Studio, OpenRouter, Together ...). Anahtar yalnızca ortam değişkeninden okunur.
* ``ollama``  — yerel Ollama sunucusu (``/api/chat``).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from turkce_rag.analysis import DEFAULT_ANALYZER
from turkce_rag.chunking import split_sentences
from turkce_rag.models import Chunk

NOT_FOUND = "Bu soruya verilen kaynaklarda bir yanıt bulamadım."

SYSTEM_PROMPT = f"""Sen Türkçe belgeler üzerinde soru yanıtlayan titiz bir asistansın.
Kurallar:
1. Yalnızca aşağıda numaralandırılmış KAYNAKLAR'daki bilgiye dayan; kendi bilgini ekleme.
2. Her cümlenin sonuna dayandığı kaynağın numarasını köşeli parantezle yaz: [1] ya da [2][3].
3. Kaynaklar soruyu yanıtlamaya yetmiyorsa yalnızca şunu yaz: "{NOT_FOUND}"
4. Kısa, açık ve doğal bir Türkçeyle yanıt ver."""

Context = tuple[int, Chunk]


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[Context]) -> str: ...


def build_user_prompt(question: str, contexts: Sequence[Context]) -> str:
    parts = ["KAYNAKLAR:"]
    for n, c in contexts:
        parts.append(f"[{n}] ({c.location()})\n{c.text}")
    parts.append(f"SORU: {question}\nYANIT:")
    return "\n\n".join(parts)


class StubLLM:
    """Ağ ve model gerektirmeyen deterministik arka uç.

    Soruyla en çok terim paylaşan en fazla ``max_sentences`` cümleyi kaynaklardan seçip atıfla
    birlikte döndürür. Üretken bir model değildir ama tüm boru hattını (atıf ayrıştırma,
    "bilmiyorum" davranışı, API/CLI çıktısı) gerçekçi biçimde uçtan uca test etmeyi sağlar.
    """

    name = "stub"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        q = set(DEFAULT_ANALYZER(question))
        if not q:
            return NOT_FOUND
        cands: list[tuple[int, int, int, str, set[str]]] = []
        for order, (n, chunk) in enumerate(contexts):
            for j, sent in enumerate(split_sentences(chunk.text)):
                cands.append((order, j, n, sent, set(DEFAULT_ANALYZER(sent))))
        # Bağlam içi IDF: birçok cümlede geçen terim ("yeraltı") az, tek cümlede geçen
        # ("giriş") çok ağırlık alır; böylece soruyu asıl yanıtlayan cümle öne çıkar.
        df = Counter(t for *_, terms in cands for t in terms & q)
        weight = {t: math.log(1 + len(cands) / df[t]) for t in df}
        total = sum(weight.values()) or 1.0
        scored: list[tuple[float, int, int, str]] = []
        for order, j, n, sent, terms in cands:
            # Arama sırası önsel bilgisi: üst sıradaki kaynağın cümlesi hafifçe tercih edilir.
            score = sum(weight[t] for t in terms & q) / total / (1 + 0.25 * order)
            if score > 0:
                # Eşitlikte üst sıradaki kaynak ve önceki cümle kazanır.
                scored.append((score, -order, -j, f"{sent} [{n}]"))
        if not scored:
            return NOT_FOUND
        scored.sort(reverse=True)
        best = scored[0][0]
        # Ek cümleler yalnızca en iyisi kadar ilgiliyse eklenir (yarısından az değil).
        picked = [s for score, *_, s in scored[: self.max_sentences] if score >= best / 2]
        return " ".join(picked)


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("TURKCE_RAG_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY ortam değişkeni tanımlı değil")
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"openai:{self.model}"

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts)},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()


class OllamaLLM:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.environ.get("TURKCE_RAG_OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def generate(self, question: str, contexts: Sequence[Context]) -> str:
        import httpx

        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": self.temperature},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, contexts)},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return str(resp.json()["message"]["content"]).strip()


def make_llm(spec: str | None) -> LLM:
    """``stub`` | ``openai[:model]`` | ``ollama[:model]``."""
    spec = (spec or os.environ.get("TURKCE_RAG_LLM", "stub")).strip()
    kind, _, model = spec.partition(":")
    kind = kind.lower()
    if kind == "stub":
        return StubLLM()
    if kind == "openai":
        return OpenAICompatibleLLM(model=model or None)
    if kind == "ollama":
        return OllamaLLM(model=model or None)
    raise ValueError(f"Bilinmeyen LLM arka ucu: {spec!r} (stub | openai[:model] | ollama[:model])")


_CITATION = re.compile(r"\[(\d{1,2})\]")


def parse_citations(text: str, valid: Sequence[int]) -> list[int]:
    """Cevaptaki ``[n]`` atıflarını ilk geçiş sırasıyla, yalnızca geçerli numaraları döndürür."""
    allowed = set(valid)
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        if n in allowed and n not in seen:
            seen.append(n)
    return seen
