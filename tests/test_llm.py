from typing import Any

import pytest

from turkce_rag import llm as llm_mod
from turkce_rag.llm import OllamaLLM, OpenAICompatibleLLM, StubLLM, build_user_prompt, make_llm
from turkce_rag.models import Chunk

CTX = [(1, Chunk("c1", "d", "Belge", "b.md", "Kahve Yemen üzerinden geldi.", section="Tarih"))]


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def test_prompt_numbers_sources() -> None:
    p = build_user_prompt("Kahve nereden geldi?", CTX)
    assert "[1] (Belge › Tarih)" in p and p.rstrip().endswith("YANIT:")


def test_make_llm() -> None:
    assert isinstance(make_llm("stub"), StubLLM)
    with pytest.raises(ValueError):
        make_llm("bilinmeyen")


def test_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatibleLLM()


def test_openai_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-anahtar")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://yerel:8000/v1/")
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _Resp:
        seen.update(url=url, **kw)
        return _Resp({"choices": [{"message": {"content": " Yemen'den [1]. "}}]})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    out = make_llm("openai:model-x").generate("Kahve nereden geldi?", CTX)
    assert out == "Yemen'den [1]."
    assert seen["url"] == "http://yerel:8000/v1/chat/completions"
    assert seen["json"]["model"] == "model-x"
    assert seen["headers"]["Authorization"] == "Bearer test-anahtar"
    assert seen["json"]["messages"][0]["content"] == llm_mod.SYSTEM_PROMPT


def test_ollama_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _Resp:
        seen.update(url=url, **kw)
        return _Resp({"message": {"content": "Yemen [1]"}})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    model = OllamaLLM(model="qwen2.5:7b")
    assert model.generate("soru", CTX) == "Yemen [1]"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["json"]["stream"] is False


def test_stub_is_deterministic() -> None:
    s = StubLLM()
    assert s.generate("Kahve nereden geldi?", CTX) == s.generate("Kahve nereden geldi?", CTX)
    assert s.generate("Kahve nereden geldi?", CTX).endswith("[1]")
