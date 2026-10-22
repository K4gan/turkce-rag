import json
from pathlib import Path

import pytest

from turkce_rag.cli import main


@pytest.fixture()
def index_dir(tmp_path: Path, sample_dir: Path) -> Path:
    idx = tmp_path / "idx"
    assert main(["ingest", str(sample_dir), "--index", str(idx), "--embedder", "hash"]) == 0
    return idx


def test_ingest_creates_index(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    meta = json.loads((index_dir / "meta.json").read_text("utf-8"))
    assert meta["embedder"] == "hashing-512"
    assert "kapadokya" in meta["documents"]


def test_ask_text_and_json(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = ["--index", str(index_dir), "--embedder", "hash", "--llm", "stub"]
    capsys.readouterr()
    assert main(["ask", "Göreme ne zaman Dünya Mirası oldu?", *common]) == 0
    out = capsys.readouterr().out
    assert "1985" in out and "Kaynaklar:" in out and "Kapadokya" in out

    assert main(["ask", "Göreme ne zaman Dünya Mirası oldu?", "--json", *common]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["confident"] is True
    assert data["sources"][0]["doc_id"] == "kapadokya"


def test_ask_unknown_exit_code(index_dir: Path) -> None:
    rc = main(["ask", "Ankara'nın nüfusu kaçtır?", "--index", str(index_dir), "--embedder", "hash"])
    assert rc == 2


def test_search_bm25(index_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    args = ["search", "yeraltı şehirleri", "--mode", "bm25", "--index", str(index_dir)]
    assert main([*args, "--embedder", "hash"]) == 0
    assert "Kapadokya" in capsys.readouterr().out.splitlines()[0]


def test_missing_index_is_friendly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "soru", "--index", str(tmp_path / "yok"), "--embedder", "hash"]) == 1
    assert "turkce-rag ingest" in capsys.readouterr().err
