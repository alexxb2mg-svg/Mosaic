"""Benchmark Mosaic vs BM25 : Recall@10, MRR, latence. Verdict du critère de succès v1/v1.1."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from bm25 import BM25

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic import ingest
from mosaic.cli import (
    _parse_doc_weight,
    _parse_int_nonnegative,
    _parse_profile_weighting,
    _parse_rerank_depth,
    _parse_rerank_lambda,
    _parse_weights,
)
from mosaic.docio import _read_tokens
from mosaic.index import (
    EXCLUDED_DIRS,
    PROFILE_WEIGHTING_DEFAULT,
    SMOOTHING_RANK_DEFAULT,
    Index,
)
from mosaic.lexicon import load_lexicon
from mosaic.tokenize import tokenize

_EXTS = {".md", ".txt"}

ECHECS_REELS_PATH = Path(__file__).resolve().parent / "echecs_reels.jsonl"


def _charger_echecs_reels(path: Path) -> list[dict]:
    """Lit bench/echecs_reels.jsonl (banc vivant, v1.6 §F) : chaque ligne
    `{"query", "attendu", "constate"?, "date"?}` devient une requête de banc
    `{"query", "relevant", "type": "reel"}` (`attendu` -> `relevant`, promu en liste si
    fourni comme chaîne unique). Une ligne malformée (JSON invalide, `query`/`attendu`
    absent ou vide) est ignorée avec un avertissement sur stderr plutôt que de faire
    planter le banc entier — un incident réel mal formaté ne doit jamais bloquer une
    exécution de banc."""
    echecs = []
    for numero, ligne in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not ligne.strip():
            continue
        try:
            obj = json.loads(ligne)
        except json.JSONDecodeError as exc:
            print(
                f"echecs_reels.jsonl:{numero} : ligne ignorée (JSON invalide) — {exc}",
                file=sys.stderr,
            )
            continue
        query = obj.get("query")
        attendu = obj.get("attendu")
        if isinstance(attendu, str):
            attendu = [attendu]
        if not query or not isinstance(attendu, list) or not attendu:
            print(
                f"echecs_reels.jsonl:{numero} : ligne ignorée "
                f'("query" et "attendu" (liste non vide) requis) — {ligne!r}',
                file=sys.stderr,
            )
            continue
        echecs.append({"query": query, "relevant": attendu, "type": "reel"})
    return echecs


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
    parser = argparse.ArgumentParser(prog="run_bench")
    parser.add_argument("corpus")
    parser.add_argument("queries")
    parser.add_argument(
        "--config", default="v1", help="nom de la configuration (affichage)"
    )
    parser.add_argument("--embeddings", default=None, help="chemin table .msee")
    parser.add_argument(
        "--lexicon-extra", default=None, help="chemin JSON lexique additionnel"
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="a,b,g (ex. 0.25,0.15,0.60), défaut WEIGHTS_DEFAULT",
    )
    parser.add_argument(
        "--profile-weighting",
        default=PROFILE_WEIGHTING_DEFAULT,
        help=f"brut ou ppmi, défaut {PROFILE_WEIGHTING_DEFAULT}",
    )
    parser.add_argument(
        "--smoothing-rank",
        type=int,
        default=SMOOTHING_RANK_DEFAULT,
        help=f"entier >= 0, défaut {SMOOTHING_RANK_DEFAULT}",
    )
    parser.add_argument("--abtt", type=int, default=0, help="entier >= 0, défaut 0")
    parser.add_argument(
        "--doc-weight",
        type=float,
        default=None,
        help="δ, poids du canal document dans [0, 1), défaut None (non transmis -> v1.2 exact)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="active le repêcheur (v1.4) : construit rerank.msrv au build "
        "(--rerank-vectors implicite) et l'applique aux recherches Mosaic (jamais BM25)",
    )
    parser.add_argument(
        "--rerank-lambda",
        type=float,
        default=None,
        help="λ du mélange rerank, défaut moteur (0.70) si omis",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=None,
        help="profondeur du rerank, défaut moteur (50) si omis",
    )
    parser.add_argument(
        "--no-path-tokens",
        action="store_true",
        help="n'indexe pas les tokens du chemin de fichier (même drapeau que la CLI) — "
        "indispensable quand les noms de fichiers sont opaques (ex. UUID Alloprof) : "
        "les indexer injecterait du bruit hexadécimal dans les grilles et le vocabulaire",
    )
    parser.add_argument(
        "--avec-echecs",
        action="store_true",
        help="ajoute les requêtes de bench/echecs_reels.jsonl (banc vivant, v1.6 §F, type "
        '"reel") au jeu de requêtes — silencieux si le fichier est absent ou vide',
    )
    parser.add_argument(
        "--echecs-path",
        default=None,
        help="chemin alternatif à echecs_reels.jsonl (défaut : bench/echecs_reels.jsonl, "
        "à côté de ce script) — diagnostic/tests",
    )
    args = parser.parse_args()

    corpus = Path(args.corpus)

    # Garde corpus mixte : Mosaic ingère pdf/docx/xlsx/html/pptx (via markitdown) et, avec
    # --ocr, des photos (IMAGE_EXTS, v1.6 §B) — mais BM25 n'est nourri ci-dessous que des
    # .md/.txt (_EXTS). Un corpus contenant des fichiers convertibles OU des images ferait
    # comparer Mosaic (corpus complet) à BM25 (sous-ensemble) : verdict biaisé, pas un vrai
    # comparatif. Refus net plutôt qu'un chiffre trompeur. IMAGE_EXTS n'est jamais fusionné
    # dans CONVERTIBLE_EXTS (v1.6 §B) : cette garde les prend en compte explicitement.
    #
    # Revue finale v1.6 (Important, #8) : EXCLUDED_DIRS (_MOSAIC/_backups/_corbeille/
    # _cimetiere/poubelleClaude) honoré ICI aussi — Index.build() les exclut déjà de ce
    # qu'il indexe réellement ; un convertible/une image posé dedans (backup, corbeille…)
    # ne fait jamais partie du corpus comparé, le compter serait un faux positif.
    convertibles = sorted(
        p
        for p in corpus.rglob("*")
        if (
            p.suffix.lower() in ingest.CONVERTIBLE_EXTS
            or p.suffix.lower() in ingest.IMAGE_EXTS
        )
        and not (EXCLUDED_DIRS & set(p.relative_to(corpus).parts))
    )
    if convertibles:
        print(
            json.dumps(
                {
                    "error": f"corpus mixte : {len(convertibles)} fichiers convertibles détectés — "
                    "le banc compare des univers différents ; corpus .md/.txt uniquement",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        sys.exit(2)

    queries = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.avec_echecs:
        echecs_path = Path(args.echecs_path) if args.echecs_path else ECHECS_REELS_PATH
        if echecs_path.exists() and echecs_path.stat().st_size > 0:
            queries += _charger_echecs_reels(echecs_path)
    # Même exclusion que la garde ci-dessus (EXCLUDED_DIRS) : le corpus BM25 doit rester le
    # MÊME ensemble de documents que celui réellement indexé par Index.build() (qui les
    # exclut déjà) — sinon un .md dans _backups/ serait vu par BM25 mais jamais par Mosaic.
    files = sorted(
        p
        for p in corpus.rglob("*")
        if p.suffix.lower() in _EXTS
        and not (EXCLUDED_DIRS & set(p.relative_to(corpus).parts))
    )
    ids = [p.relative_to(corpus).as_posix() for p in files]

    lexicon = (
        load_lexicon(extra=Path(args.lexicon_extra)) if args.lexicon_extra else None
    )
    build_kwargs: dict = {}
    if lexicon is not None:
        build_kwargs["lexicon"] = lexicon
    if args.embeddings:
        # Pass embeddings_path + abtt; let Index.build load with correct abtt internally
        build_kwargs["embeddings_path"] = Path(args.embeddings)
    if args.weights:
        build_kwargs["weights"] = _parse_weights(args.weights)
    # Toujours transmis (contrairement à --weights/--lexicon/--embeddings) : ces trois options
    # ont désormais des défauts argparse non ambigus (== Index.build). Un filtrage
    # "seulement si != ancien défaut" ferait silencieusement retomber --profile-weighting brut
    # / --smoothing-rank 0 / --abtt 0 sur les nouveaux défauts ppmi/300 dès que la valeur
    # demandée coïncide avec l'ancienne valeur par défaut — piège de plomberie v1.2.
    build_kwargs["profile_weighting"] = _parse_profile_weighting(args.profile_weighting)
    build_kwargs["smoothing_rank"] = _parse_int_nonnegative(
        args.smoothing_rank, "smoothing-rank"
    )
    build_kwargs["abtt"] = _parse_int_nonnegative(args.abtt, "abtt")
    # --doc-weight : défaut argparse None (contrairement à --profile-weighting/--smoothing-rank/
    # --abtt ci-dessus) -> transmis SEULEMENT si explicitement fourni, pour garder le contrôle
    # v1.2 exact par défaut (Index.build(doc_weight=0.0) est déjà ce défaut, mais un None
    # explicite documente l'intention : ce banc n'a pas d'opinion sur δ tant qu'on ne le lui
    # demande pas).
    if args.doc_weight is not None:
        build_kwargs["doc_weight"] = _parse_doc_weight(args.doc_weight)
    if args.no_path_tokens:
        build_kwargs["index_paths"] = False
    # --rerank implique --rerank-vectors au build : sans rerank.msrv, idx.search(rerank=True)
    # échouerait — le banc n'a pas de sens à demander la recherche sans le construire.
    # --rerank-lambda/--rerank-depth validés MAINTENANT (avant le build, potentiellement long) :
    # même contrat que cli.py, échec rapide plutôt qu'après avoir payé tout le coût du build.
    search_kwargs: dict = {}
    if args.rerank:
        build_kwargs["rerank_vectors"] = True
        search_kwargs["rerank"] = True
        if args.rerank_lambda is not None:
            search_kwargs["rerank_lambda"] = _parse_rerank_lambda(args.rerank_lambda)
        if args.rerank_depth is not None:
            search_kwargs["rerank_depth"] = _parse_rerank_depth(args.rerank_depth)

    t_build = time.perf_counter()
    idx = Index.build(
        corpus, corpus.parent / f"_bench_index_{args.config}", **build_kwargs
    )
    build_s = round(time.perf_counter() - t_build, 1)
    bm25 = BM25([_read_tokens(p) for p in files])
    # --rerank : toujours transmis à idx.search si demandé (jamais à BM25, qui n'a pas de
    # notion de rerank potion).

    report: dict = {"config": args.config, "build_s": build_s}
    for name in ("mosaic", "bm25"):
        results, times = [], []
        for q in queries:
            t0 = time.perf_counter()
            if name == "mosaic":
                hits = [h["id"] for h in idx.search(q["query"], k=10, **search_kwargs)]
            else:
                hits = [ids[i] for i in bm25.search(tokenize(q["query"]), k=10)]
            times.append((time.perf_counter() - t0) * 1000)
            results.append(hits)
        entry = _metrics(results, [q["relevant"] for q in queries])
        entry["latence_ms_mediane"] = round(statistics.median(times), 2)
        for qtype in ("lexical", "semantique", "reel"):
            sel = [i for i, q in enumerate(queries) if q.get("type") == qtype]
            if sel:
                entry[qtype] = _metrics(
                    [results[i] for i in sel], [queries[i]["relevant"] for i in sel]
                )
        report[name] = entry

    # verdict_v1 n'a de sens que si le jeu de requêtes distingue les pièges sémantiques
    # (type "semantique") : sur un jeu non typé (ex. bench/alloprof.py), la clause
    # sémantique comparerait 0 > 0 et rendrait un verdict faussement négatif — on
    # l'omet alors plutôt que de publier un faux échec.
    if any(q.get("type") == "semantique" for q in queries):
        report["verdict_v1"] = report["mosaic"]["recall@10"] >= report["bm25"][
            "recall@10"
        ] and report["mosaic"].get("semantique", {}).get("recall@10", 0) > report[
            "bm25"
        ].get("semantique", {}).get("recall@10", 0)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
