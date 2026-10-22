import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from turkce_rag.api import create_app
from turkce_rag.pipeline import RAGEngine


@pytest.fixture()
def client(engine: RAGEngine) -> TestClient:
    return TestClient(create_app(engine=engine))


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["chunks"] > 0 and body["llm"] == "stub"


def test_ask(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "Yıllık eğitim bütçesi ne kadar?"})
    assert r.status_code == 200
    body = r.json()
    assert body["confident"] is True
    assert "12.000 TL" in body["answer"]
    assert body["sources"][0]["page"] == 3


def test_search_and_validation(client: TestClient) -> None:
    r = client.post("/search", json={"query": "RuBisCO", "top_k": 2, "mode": "bm25"})
    assert r.status_code == 200 and r.json()[0]["chunk"]["doc_id"] == "fotosentez"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "yok"}).status_code == 422


def test_ingest_rejects_local_paths(client: TestClient) -> None:
    r = client.post("/ingest", json={"urls": ["/etc/passwd"]})
    assert r.status_code == 400
