"""Étape 1-bis du chantier atlas (briefing #367) — l'atlas comme CANAL DE RAPPEL de la fusion.

L'étape 1 a établi le profil : sur Alloprof, l'atlas 64×64 bat la grille en RAPPEL
(R@10 0.6233 vs 0.5567) mais reste derrière en précision top-1 — un candidat canal de
rappel, pas un remplaçant. La carte des terrains (README) a établi que la fusion RRF
gagne sur terrain lexical (Alloprof) et dilue ailleurs. La question jointe :

  AJOUTER le canal atlas au trio RRF (grille + BM25 + embeddings) améliore-t-il le
  rappel sur LE terrain où la fusion est déjà la bonne architecture ?

PRÉDICTION DÉCLARÉE AVANT MESURE (falsifiable) : le canal est ADOPTÉ dans le dossier
#367 si trio+atlas > trio d'au moins +1 pt de recall@10 ; sinon ligne de carte «
écartée sur ce terrain » (les erreurs de l'atlas seraient corrélées à celles de la
grille — même corpus appris — et la fusion n'aurait rien à décorréler).

PROTOCOLE : échantillon Alloprof 500 docs / 300 requêtes réelles (conventions de
l'étape 1), quatre canaux en classements profondeur 50 :
- grille : Index.build config calibrée Alloprof (poids 0.50/0.30/0.20,
  --no-path-tokens, embeddings potion + abtt 2 — la config du banc public) ;
- BM25 : sur le MÊME flux de tokens que la grille (canonicalisation + collocations) ;
- embeddings : potion-multilingual-128M plein texte (via mosaic.rerank) ;
- atlas : cartes de chaleur tf×idf sur SOM 64×64, σ=0 (leçon étape 1 : la convolution
  n'aide jamais sur terrain hostile), cosinus plein corpus.
RRF K=60. Rapporte chaque canal seul + trio + trio+atlas (+ duo grille+atlas témoin).

Usage : python research/atlas_fusion.py <corpus_alloprof> <verite.jsonl> <table.msee>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault("MOSAIC_ATLAS", "64")
_ARGS = sys.argv[1:4]
sys.argv = sys.argv[:1]  # atlas_som lit argv à l'import

import json  # noqa: E402
import tempfile  # noqa: E402

from atlas_som import (  # noqa: E402
    ATLAS,
    DIM_SOM,
    ECHANTILLON_DOCS,
    GRID_BUILD,
    MAX_REQUETES,
    _carte,
    _projeter,
    _som,
)
from mosaic import rerank as rerank_module  # noqa: E402
from mosaic.calibration import _preparer  # noqa: E402
from mosaic.collocations import merge  # noqa: E402
from mosaic.index import Index  # noqa: E402
from mosaic.lexicon import canonicalize, compile_lexicon  # noqa: E402
from mosaic.tokenize import tokenize  # noqa: E402

PROFONDEUR = 50
K_RRF = 60
POIDS_ALLOPROF = (0.50, 0.30, 0.20)  # calibration mesurée du banc public


def _metriques(classements: list[list[str]], verites: list[list[str]]) -> dict:
    recalls, rrs = [], []
    for cls, rel in zip(classements, verites):
        rel_set = set(rel)
        recalls.append(len(rel_set & set(cls[:10])) / max(1, len(rel_set)))
        rr = next((1.0 / r for r, d in enumerate(cls, start=1) if d in rel_set), 0.0)
        rrs.append(rr)
    n = max(1, len(classements))
    return {
        "recall@10": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
    }


def _rrf(*classements: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for cls in classements:
        for rang, doc in enumerate(cls, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (K_RRF + rang)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:10]


def main() -> int:
    if len(_ARGS) < 3:
        raise SystemExit(
            "usage : python research/atlas_fusion.py <corpus> <verite.jsonl> <table.msee>"
        )
    corpus_src, verite_path, table = Path(_ARGS[0]), Path(_ARGS[1]), Path(_ARGS[2])
    for p in (corpus_src, verite_path, table):
        if not p.exists():
            raise SystemExit(f"introuvable : {p}")
    if not rerank_module.available():
        raise SystemExit('model2vec requis — pip install "model2vec==0.8.2"')

    requetes = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(line)
            for line in verite_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    # échantillon (mêmes conventions déterministes que les étapes 1 et 2)
    fichiers = sorted(p for p in corpus_src.iterdir() if p.is_file())
    dossier = tempfile.TemporaryDirectory()
    corpus = corpus_src
    if len(fichiers) > ECHANTILLON_DOCS:
        corpus = Path(dossier.name) / "corpus"
        corpus.mkdir()
        garde = fichiers[:ECHANTILLON_DOCS]
        for p in garde:
            (corpus / p.name).write_bytes(p.read_bytes())
        noms = {p.name for p in garde}
        requetes = [(q, rel) for q, rel in requetes if all(r in noms for r in rel)][
            :MAX_REQUETES
        ]
    verites = [rel for _q, rel in requetes]
    print(
        f"corpus {corpus_src.name} : échantillon {ECHANTILLON_DOCS}, "
        f"{len(requetes)} requêtes"
    )

    docs, profiles, colloc, lexicon = _preparer(corpus, None, 300, GRID_BUILD)
    ids = [d for d, _ in docs]
    compiled = compile_lexicon(lexicon)

    def tokens_requete(q: str) -> list[str]:
        return merge(merge(canonicalize(tokenize(q), compiled), colloc), colloc)

    # --- canal grille : la config calibrée du banc public ---
    with tempfile.TemporaryDirectory() as tmp:
        idx = Index.build(
            corpus,
            Path(tmp) / "idx",
            weights=POIDS_ALLOPROF,
            index_paths=False,
            embeddings_path=table,
            abtt=2,
        )
        idx.chauffer_recherche()
        cls_grille = [
            [h["id"] for h in idx.search(q, k=PROFONDEUR)] for q, _rel in requetes
        ]

    # --- canal BM25 : même flux de tokens que la grille ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    from bm25 import BM25

    bm = BM25([toks for _d, toks in docs])
    cls_bm25 = [
        [ids[i] for i in bm.search(tokens_requete(q), k=PROFONDEUR)]
        for q, _rel in requetes
    ]

    # --- canal embeddings : potion plein texte ---
    textes = [(corpus / d).read_text(encoding="utf-8", errors="replace") for d in ids]
    mat = rerank_module.encode_texts(textes)
    cls_m2v = []
    for q, _rel in requetes:
        qv = rerank_module.encode_query(q)
        cls_m2v.append([ids[i] for i in np.argsort(-(mat @ qv))[:PROFONDEUR]])

    # --- canal atlas : cartes de chaleur sur SOM, σ=0 ---
    v = len(profiles.rows)
    features = _projeter(profiles.acc[:v], DIM_SOM)
    bmu = _som(features)
    position = {t: int(bmu[i]) for t, i in profiles.rows.items()}
    h, w = ATLAS
    cartes_docs = np.stack(
        [_carte(toks, position, profiles.idf, h, w) for _d, toks in docs]
    )
    plat = cartes_docs.reshape(len(ids), -1)
    normes = np.linalg.norm(plat, axis=1)
    normes[normes == 0] = 1.0
    cls_atlas = []
    for q, _rel in requetes:
        cq = _carte(tokens_requete(q), position, profiles.idf, h, w).ravel()
        nq = np.linalg.norm(cq)
        scores = (plat @ cq) / (normes * (nq if nq else 1.0))
        cls_atlas.append(
            [ids[i] for i in np.argsort(-scores, kind="stable")[:PROFONDEUR]]
        )

    canaux = {
        "grille (calibrée)": cls_grille,
        "bm25": cls_bm25,
        "m2v": cls_m2v,
        "atlas (σ=0)": cls_atlas,
    }
    fusions = {
        "trio (grille+bm25+m2v)": [
            _rrf(a, b, c) for a, b, c in zip(cls_grille, cls_bm25, cls_m2v)
        ],
        "trio + atlas": [
            _rrf(a, b, c, d)
            for a, b, c, d in zip(cls_grille, cls_bm25, cls_m2v, cls_atlas)
        ],
        "grille + atlas (témoin)": [_rrf(a, d) for a, d in zip(cls_grille, cls_atlas)],
    }
    for nom, cls in list(canaux.items()) + list(fusions.items()):
        m = _metriques(cls, verites)
        print(f"{nom:<26} R@10 {m['recall@10']:<8} MRR {m['mrr']}")
    print("\nPrédiction (en-tête) : canal ADOPTÉ si trio+atlas >= trio +1 pt R@10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
