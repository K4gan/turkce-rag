"""Komut satırı arayüzü: ``turkce-rag ingest | ask | search | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

DEFAULT_INDEX = os.environ.get("TURKCE_RAG_INDEX", ".turkce-rag")
DEFAULT_EMBEDDER = os.environ.get("TURKCE_RAG_EMBEDDER", "e5")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--index", default=DEFAULT_INDEX, help=f"indeks klasörü (vars.: {DEFAULT_INDEX})"
    )
    p.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help="e5 | hash | none | <HF model adı> (vars.: e5 = intfloat/multilingual-e5-small)",
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    from turkce_rag.chunking import ChunkConfig
    from turkce_rag.embeddings import make_embedder
    from turkce_rag.index import Index
    from turkce_rag.pipeline import RAGEngine

    try:
        index = Index.load(args.index)
    except FileNotFoundError:
        index = Index()
    engine = RAGEngine(
        index=index,
        embedder=make_embedder(args.embedder),
        chunk_config=ChunkConfig(max_chars=args.max_chars),
    )
    added = engine.ingest(args.inputs)
    engine.save(args.index)
    print(
        f"✓ {added} parça eklendi · indekste toplam {len(engine.index)} parça, "
        f"{len(engine.index.doc_ids())} belge → {args.index}"
    )
    return 0


def _engine(args: argparse.Namespace):
    from turkce_rag.pipeline import AnswerPolicy, RAGEngine

    return RAGEngine.open(
        args.index,
        embedder=args.embedder,
        llm=getattr(args, "llm", None),
        policy=AnswerPolicy(min_coverage=getattr(args, "min_coverage", 0.5)),
    )


def _cmd_ask(args: argparse.Namespace) -> int:
    engine = _engine(args)
    answer = engine.ask(args.question, k=args.top_k, mode=args.mode)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(answer.text)
    if answer.sources:
        print("\nKaynaklar:")
        for s in answer.sources:
            print(f"  [{s.n}] {s.location}  ({s.source})")
    return 0 if answer.confident else 2


def _cmd_search(args: argparse.Namespace) -> int:
    engine = _engine(args)
    hits = engine.search(args.query, k=args.top_k, mode=args.mode)
    for r, h in enumerate(hits, start=1):
        snippet = h.chunk.text[:160].replace("\n", " ")
        print(f"{r:>2}. {h.score:.4f}  {h.chunk.location()}\n    {snippet}…")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("API için: pip install 'turkce-rag[api]'", file=sys.stderr)
        return 1
    from turkce_rag.api import create_app

    app = create_app(index_dir=args.index, embedder=args.embedder, llm=args.llm)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turkce-rag", description="Türkçe öncelikli, kaynak gösteren hibrit RAG motoru."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="dosya, klasör veya URL indeksle")
    p.add_argument("inputs", nargs="+", help="PDF/MD/TXT dosyası, klasör ya da http(s) URL")
    p.add_argument("--max-chars", type=int, default=700, help="parça başına en fazla karakter")
    _add_common(p)
    p.set_defaults(func=_cmd_ingest)

    for name, helptext in (("ask", "soru sor (atıflı yanıt)"), ("search", "yalnızca arama")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("question" if name == "ask" else "query")
        p.add_argument("-k", "--top-k", type=int, default=4)
        p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=None)
        _add_common(p)
        if name == "ask":
            p.add_argument("--llm", default=None, help="stub | openai[:model] | ollama[:model]")
            p.add_argument("--min-coverage", type=float, default=0.5)
            p.add_argument("--json", action="store_true", help="JSON çıktı")
            p.set_defaults(func=_cmd_ask)
        else:
            p.set_defaults(func=_cmd_search)

    p = sub.add_parser("serve", help="HTTP API'yi başlat")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--llm", default=None)
    _add_common(p)
    p.set_defaults(func=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
