"""Örnek derlem üzerinde arama ve cevaplama değerlendirmesi.

    python eval/run_eval.py                 # e5 + hash, tüm yapılandırmalar
    python eval/run_eval.py --no-e5         # model indirmeden (yalnızca BM25 + hash)

Sonuçlar ekrana Markdown tablo olarak basılır ve ``eval/results.json`` dosyasına yazılır.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

from turkce_rag.analysis import Analyzer
from turkce_rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from turkce_rag.evaluation import evaluate_retrieval, load_qa
from turkce_rag.index import Index
from turkce_rag.loaders import iter_sources
from turkce_rag.normalize import fold_diacritics
from turkce_rag.pipeline import AnswerPolicy, RAGEngine
from turkce_rag.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--qa", default=str(ROOT / "eval" / "qa.jsonl"))
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--no-e5", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = ap.parse_args()

    items = load_qa(args.qa)
    # Aynı sorular, Türkçe karakterleri olmayan klavyeyle yazılmış gibi: "Göreme" → "Goreme".
    ascii_items = [replace(it, question=fold_diacritics(it.question)) for it in items]
    docs = list(iter_sources([args.corpus]))

    embedders = [HashingEmbedder()]
    if not args.no_e5:
        t0 = time.perf_counter()
        embedders.insert(0, SentenceTransformerEmbedder(args.model))
        print(f"model yüklendi: {args.model} ({time.perf_counter() - t0:.1f}s)")

    reports = []
    ascii_reports = []
    # 1) BM25: dil bilgisi olmayan taban çizgisi vs. Türkçe analizör
    for label, analyzer in (
        ("BM25 (düz tokenizasyon)", Analyzer.plain()),
        ("BM25 (Türkçe analizör)", Analyzer()),
    ):
        idx = Index(analyzer=analyzer)
        idx.add_documents(docs)
        r = HybridRetriever(idx)
        fn = lambda q, k, r=r: [h.chunk for h in r.search(q, k, "bm25")]  # noqa: E731
        reports.append(evaluate_retrieval(label, items, fn, KS))
        ascii_reports.append(evaluate_retrieval(label, ascii_items, fn, KS))

    engines: dict[str, RAGEngine] = {}
    for emb in embedders:
        idx = Index()
        t0 = time.perf_counter()
        idx.add_documents(docs, emb)
        build_s = time.perf_counter() - t0
        r = HybridRetriever(idx, emb)
        short = "e5-small" if "e5" in emb.name else "hash"
        print(f"{emb.name}: {len(idx)} parça, indeksleme {build_s:.1f}s")
        for mode, label in (
            ("dense", f"Dense ({short})"),
            ("hybrid", f"Hibrit RRF (BM25-TR + {short})"),
        ):
            fn = lambda q, k, r=r, m=mode: [h.chunk for h in r.search(q, k, m)]  # noqa: E731
            reports.append(evaluate_retrieval(label, items, fn, KS))
            ascii_reports.append(evaluate_retrieval(label, ascii_items, fn, KS))
        engines[short] = RAGEngine(index=idx, embedder=emb, policy=AnswerPolicy())

    n = reports[0].n
    print(f"\n### Arama ({n} cevaplanabilir soru, {len(docs)} belge, {len(idx)} parça)\n")
    print("| Yöntem | Recall@1 | Recall@3 | Recall@5 | MRR@10 | MRR@10 (ASCII sorgu) |")
    print("|---|---:|---:|---:|---:|---:|")
    for rep, arep in zip(reports, ascii_reports, strict=True):
        rc = " | ".join(f"{rep.recall[k]:.3f}" for k in KS)
        print(f"| {rep.name} | {rc} | {rep.mrr:.3f} | {arep.mrr:.3f} |")

    # 2) Cevaplama davranışı: güven eşiği + stub LLM (hibrit, en iyi gömme ile)
    key = "e5-small" if "e5-small" in engines else "hash"
    eng = engines[key]
    ans = [it for it in items if it.answerable]
    una = [it for it in items if not it.answerable]
    answered_ok = cited_ok = abstained = 0
    false_abstain: list[str] = []
    false_answer: list[str] = []
    for it in ans:
        a = eng.ask(it.question)
        if a.confident:
            answered_ok += 1
            gold = [h.chunk for h in a.hits if h.chunk.doc_id == it.doc_id]
            if any(s.doc_id == it.doc_id for s in a.sources) and gold:
                cited_ok += 1
        else:
            false_abstain.append(it.id)
    for it in una:
        a = eng.ask(it.question)
        if not a.confident:
            abstained += 1
        else:
            false_answer.append(it.id)
    print(f"\n### Cevaplama (hibrit + {key}, stub LLM, min_coverage={eng.policy.min_coverage})\n")
    print(
        f"- Cevaplanabilir sorularda yanıt verildi: {answered_ok}/{len(ans)}"
        f" (doğru belgeye atıfla: {cited_ok}/{len(ans)})"
    )
    print(f"- Cevaplanamaz sorularda 'bulamadım' dendi: {abstained}/{len(una)}")
    if false_abstain or false_answer:
        print(f"- Hatalı çekimser: {false_abstain} · hatalı yanıt: {false_answer}")

    # Eşik taraması: "bilmiyorum" kararının ödünleşimi (LLM çağrısı gerektirmez).
    print("\n| min_coverage | Cevaplanabilir → yanıtlandı | Cevaplanamaz → çekimser |")
    print("|---:|---:|---:|")
    sweep = []
    hits_by_item = {it.id: eng.search(it.question, eng.policy.context_k) for it in items}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        ok = sum(max((h.coverage for h in hits_by_item[it.id]), default=0) >= th for it in ans)
        ab = sum(max((h.coverage for h in hits_by_item[it.id]), default=0) < th for it in una)
        sweep.append({"min_coverage": th, "answered": ok, "abstained": ab})
        print(f"| {th:.1f} | {ok}/{len(ans)} | {ab}/{len(una)} |")

    out = {
        "python": platform.python_version(),
        "documents": len(docs),
        "chunks": len(idx),
        "questions_answerable": len(ans),
        "questions_unanswerable": len(una),
        "retrieval": [
            {
                "name": r.name,
                "recall": {str(k): v for k, v in r.recall.items()},
                "mrr@10": r.mrr,
                "mrr@10_ascii": a.mrr,
                "misses@5": r.misses,
            }
            for r, a in zip(reports, ascii_reports, strict=True)
        ],
        "threshold_sweep": sweep,
        "answering": {
            "embedder": key,
            "answered": answered_ok,
            "answered_with_correct_doc_citation": cited_ok,
            "abstained_on_unanswerable": abstained,
            "false_abstain": false_abstain,
            "false_answer": false_answer,
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
