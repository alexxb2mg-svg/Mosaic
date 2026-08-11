"""Baseline concurrente : embeddings statiques model2vec (potion-multilingual-128M).

Mesure la qualité de récupération d'un modèle appris sur les mêmes requêtes/corpus
que Mosaic — le juge de paix de la question « réinvente-t-on la poudre ? ».
Outillage de banc, hors produit.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from model2vec import StaticModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_EXTS = {".md", ".txt"}
MODELE = "minishlab/potion-multilingual-128M"


def _metrics(results: list[list[str]], relevant: list[list[str]]) -> dict:
    recalls, rranks = [], []
    for hits, rel in zip(results, relevant, strict=True):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(hits[:10])) / max(len(rel_set), 1))
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h in rel_set:
                rr = 1.0 / rank
                break
        rranks.append(rr)
    return {
        "recall@10": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(rranks), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="baseline_model")
    parser.add_argument("corpus")
    parser.add_argument("queries")
    parser.add_argument(
        "--details", action="store_true", help="détail par piège sémantique"
    )
    args = parser.parse_args()

    corpus = Path(args.corpus)
    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in _EXTS)
    ids = [p.relative_to(corpus).as_posix() for p in files]

    t0 = time.perf_counter()
    model = StaticModel.from_pretrained(MODELE)
    load_s = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    docs = [p.read_text(encoding="utf-8", errors="replace") for p in files]
    mat = model.encode(docs)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    build_s = round(time.perf_counter() - t0, 1)

    results, times = [], []
    for q in queries:
        t0 = time.perf_counter()
        qv = model.encode([q["query"]])[0]
        qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
        scores = mat @ qv
        top = np.argsort(-scores)[:10]
        times.append((time.perf_counter() - t0) * 1000)
        results.append([ids[i] for i in top])

    report: dict = {
        "config": f"model2vec:{MODELE}",
        "load_s": load_s,
        "build_s": build_s,
    }
    entry = _metrics(results, [q["relevant"] for q in queries])
    entry["latence_ms_mediane"] = round(statistics.median(times), 2)
    for qtype in ("lexical", "semantique"):
        sel = [i for i, q in enumerate(queries) if q.get("type") == qtype]
        if sel:
            entry[qtype] = _metrics(
                [results[i] for i in sel], [queries[i]["relevant"] for i in sel]
            )
    report["model2vec"] = entry

    if args.details:
        details = []
        for i, q in enumerate(queries):
            if q["type"] != "semantique":
                continue
            rank = next(
                (results[i].index(r) + 1 for r in q["relevant"] if r in results[i]),
                None,
            )
            details.append({"query": q["query"][:60], "rang": rank})
        report["pieges"] = details

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
