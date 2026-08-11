"""Fusions RRF : le standard du marché (BM25+model2vec) contre nos assemblages.

RRF (Reciprocal Rank Fusion, k=60) : score(doc) = Σ_systèmes 1/(k + rang_doc).
Outillage de banc, hors produit.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from bm25 import BM25
from model2vec import StaticModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.docio import _read_tokens
from mosaic.embeddings import Embeddings
from mosaic.index import Index
from mosaic.tokenize import tokenize

_EXTS = {".md", ".txt"}
K_RRF = 60
PROFONDEUR = 50


def _metrics(results: list[list[str]], relevant: list[list[str]]) -> dict:
    recalls, rranks = [], []
    for hits, rel in zip(results, relevant, strict=True):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(hits[:10])) / max(len(rel_set), 1))
        rr = next((1.0 / r for r, h in enumerate(hits, start=1) if h in rel_set), 0.0)
        rranks.append(rr)
    return {
        "recall@10": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(rranks), 4),
    }


def _rrf(*classements: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for cls in classements:
        for rang, doc in enumerate(cls, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (K_RRF + rang)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:10]


def main(
    corpus_dir: str,
    queries_path: str,
    weights: tuple[float, float, float] = (0.25, 0.15, 0.60),
    index_paths: bool = True,
) -> None:
    corpus = Path(corpus_dir)
    queries = [
        json.loads(line)
        for line in Path(queries_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    files = sorted(p for p in corpus.rglob("*") if p.suffix.lower() in _EXTS)
    ids = [p.relative_to(corpus).as_posix() for p in files]

    print("préparation des trois systèmes…", file=sys.stderr)
    bm25 = BM25([_read_tokens(p) for p in files])
    m2v = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
    docs_txt = [p.read_text(encoding="utf-8", errors="replace") for p in files]
    mat = m2v.encode(docs_txt)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    idx = Index.build(
        corpus,
        corpus.parent / "_bench_fusion_idx",
        embeddings=Embeddings.load(Path("data_externes/potion_fr.msee"), abtt=2),
        embeddings_path=Path("data_externes/potion_fr.msee"),
        abtt=2,
        weights=weights,
        index_paths=index_paths,
    )

    par_config: dict[str, list[list[str]]] = {
        c: []
        for c in (
            "bm25",
            "m2v",
            "rrf_bm25_m2v",
            "mosaic",
            "rrf_mosaic_bm25",
            "rrf_tous",
        )
    }
    for q in queries:
        cls_bm25 = [ids[i] for i in bm25.search(tokenize(q["query"]), k=PROFONDEUR)]
        qv = m2v.encode([q["query"]])[0]
        qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
        cls_m2v = [ids[i] for i in np.argsort(-(mat @ qv))[:PROFONDEUR]]
        cls_mosaic = [h["id"] for h in idx.search(q["query"], k=PROFONDEUR)]
        par_config["bm25"].append(cls_bm25[:10])
        par_config["m2v"].append(cls_m2v[:10])
        par_config["mosaic"].append(cls_mosaic[:10])
        par_config["rrf_bm25_m2v"].append(_rrf(cls_bm25, cls_m2v))
        par_config["rrf_mosaic_bm25"].append(_rrf(cls_mosaic, cls_bm25))
        par_config["rrf_tous"].append(_rrf(cls_mosaic, cls_bm25, cls_m2v))

    report: dict = {}
    relevant = [q["relevant"] for q in queries]
    for config, results in par_config.items():
        entry = _metrics(results, relevant)
        for qtype in ("lexical", "semantique"):
            sel = [i for i, q in enumerate(queries) if q.get("type") == qtype]
            # Jeu non typé (ex. bench/alloprof.py) : aucun sous-ensemble — statistics.mean
            # planterait sur une liste vide, on omet la clé plutôt que de planter le banc.
            if sel:
                entry[qtype] = _metrics(
                    [results[i] for i in sel], [relevant[i] for i in sel]
                )
        report[config] = entry
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="fusion_bench")
    parser.add_argument("corpus")
    parser.add_argument("queries")
    parser.add_argument(
        "--weights",
        default="0.25,0.15,0.60",
        help="a,b,g du canal mosaic — passer les poids CALIBRÉS du corpus mesuré, "
        "sinon la fusion juge mosaic dans une configuration qui n'est pas la sienne",
    )
    parser.add_argument(
        "--no-path-tokens",
        action="store_true",
        help="même drapeau que la CLI/run_bench : noms de fichiers opaques (ex. UUID)",
    )
    args = parser.parse_args()
    a, b, g = (float(x) for x in args.weights.split(","))
    main(
        args.corpus,
        args.queries,
        weights=(a, b, g),
        index_paths=not args.no_path_tokens,
    )
