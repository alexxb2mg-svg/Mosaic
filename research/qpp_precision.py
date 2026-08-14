"""La confiance NQC prédit-elle vraiment le succès d'une requête ?

Un indicateur de confiance qui ne corrèle pas avec le succès est pire qu'aucun :
il donne au moteur l'air de savoir ce qu'il ignore. Ce banc mesure la corrélation
sur un terrain où la vérité existe — Alloprof, 2 316 requêtes annotées.

MÉTHODE : pour chaque requête, on calcule NQC sur les scores de CHAQUE canal, et
on regarde si la requête a réussi (au moins un document pertinent dans le top 10).
La corrélation utilisée est le **point-biserial** (une variable continue contre
une binaire), qui est le coefficient de Pearson dans ce cas — pas de dépendance
supplémentaire.

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 21h40) :

  P-QPP1 — la corrélation est POSITIVE et vaut au moins 0,15 sur au moins un
           canal. La littérature donne ~0,4 en Kendall sur d'autres collections ;
           on vise plus bas parce que le corpus et le moteur diffèrent, et parce
           qu'une corrélation faible mais réelle suffit à trier.

  P-QPP2 — NQC calculé sur les scores de FUSION sera peu informatif : le RRF ne
           rend pas des similarités mais des sommes de 1/(60+rang), qui écrasent
           les amplitudes par construction. Si cette prédiction tient, la
           confiance devra se lire sur un CANAL, jamais sur la fusion — c'est une
           conséquence de conception, pas un détail.

  P-QPP3 — le biais nommé par la littérature se verra : NQC corrélera aussi avec
           la LONGUEUR de la requête. On le mesure explicitement pour savoir si
           l'indicateur mesure la difficulté ou le nombre de mots.

CRITÈRE D'ADOPTION, déclaré avant : P-QPP1 tenue sur au moins un canal, ET la
corrélation avec le succès supérieure à celle avec la longueur (sinon l'indicateur
mesure la requête, pas le corpus). Sinon : documenté et écarté.

Usage : python research/qpp_precision.py [--echantillon 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.confiance import nqc  # noqa: E402
from mosaic.index import Index  # noqa: E402

CORPUS = Path(os.environ.get("MOSAIC_BENCH_CORPUS", RACINE / "bench/alloprof/corpus"))
VERITE = Path(
    os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/alloprof/verite.jsonl")
)
POTION = Path(
    os.environ.get("MOSAIC_POTION", RACINE / "data_externes/potion_fr_abtt2.msee")
)


def point_biserial(continu: list[float], binaire: list[bool]) -> float:
    """Corrélation entre une variable continue et une binaire (= Pearson ici)."""
    x = np.asarray(continu, dtype=np.float64)
    y = np.asarray(binaire, dtype=np.float64)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--echantillon", type=int, default=0, help="0 = toutes")
    args = ap.parse_args()

    requetes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    if args.echantillon:
        requetes = requetes[: args.echantillon]

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        idx = Index.build(
            CORPUS,
            Path(tmp) / "idx",
            embeddings_path=POTION,
            abtt=2,
            weights=(0.25, 0.15, 0.60),
            rerank_vectors=True,
            type_doc=True,
            hybride=True,
            atlas=True,
        )
        print(
            f"index construit en {time.perf_counter() - t0:.0f}s ({len(idx.ids)} docs)",
            flush=True,
        )

        # On rejoue les canaux À LA MAIN pour disposer des scores COMPLETS
        # (search ne rend que le top-k ; NQC a besoin de toute la distribution).
        from mosaic import rerank as rerank_module
        from mosaic.docio import tokenize
        from mosaic.lexicon import canonicalize
        from mosaic.queries import _cos_all, merge

        confiances: dict[str, list[float]] = {"grille": [], "bm25": [], "embed": []}
        succes: list[bool] = []
        longueurs: list[float] = []

        for n, q in enumerate(requetes, 1):
            texte = q["query"]
            pertinents = set(q["relevant"])
            hits = idx.search(texte, k=10, fusion=True)
            succes.append(bool({h["id"] for h in hits} & pertinents))
            longueurs.append(float(len(texte.split())))

            confiances["grille"].append(nqc(_cos_all(idx, texte)))
            toks = merge(
                merge(canonicalize(tokenize(texte), idx._compiled), idx.colloc),
                idx.colloc,
            )
            assert idx.bm25 is not None  # index construit --hybride, garde ci-dessus
            confiances["bm25"].append(nqc(idx.bm25.scores(toks)))
            confiances["embed"].append(
                nqc(idx.rerank_vecs @ rerank_module.encode_query(texte))
            )
            if n % 500 == 0:
                print(f"  {n}/{len(requetes)}", flush=True)

    taux = 100 * sum(succes) / max(1, len(succes))
    print(
        f"\n{len(succes)} requêtes, {taux:.1f} % réussies (au moins 1 pertinent au top 10)"
    )
    print(f"\n{'canal':10s} {'corr. succès':>13s} {'corr. longueur':>15s}")
    resultats = {}
    for canal, valeurs in confiances.items():
        c_succes = point_biserial(valeurs, succes)
        c_long = (
            float(np.corrcoef(valeurs, longueurs)[0, 1]) if np.std(valeurs) else 0.0
        )
        resultats[canal] = {
            "corr_succes": round(c_succes, 4),
            "corr_longueur": round(c_long, 4),
        }
        print(f"{canal:10s} {c_succes:13.4f} {c_long:15.4f}")

    meilleur = max(resultats, key=lambda c: resultats[c]["corr_succes"])
    m = resultats[meilleur]
    print(
        f"\nP-QPP1  meilleur canal « {meilleur} » : {m['corr_succes']:+.4f} -> "
        + ("TENUE" if m["corr_succes"] >= 0.15 else "FAUSSE")
    )
    print(
        f"P-QPP3  corr. longueur {m['corr_longueur']:+.4f} vs succès "
        f"{m['corr_succes']:+.4f} -> "
        + (
            "le signal est la DIFFICULTÉ"
            if abs(m["corr_succes"]) > abs(m["corr_longueur"])
            else "le signal est la LONGUEUR (biais confirmé)"
        )
    )
    verdict = (
        "ADOPTÉ"
        if m["corr_succes"] >= 0.15 and abs(m["corr_succes"]) > abs(m["corr_longueur"])
        else "ÉCARTÉ"
    )
    print(f"CRITÈRE -> {verdict}")

    Path(__file__).with_name("resultats_qpp.json").write_text(
        json.dumps(
            {
                "requetes": len(succes),
                "taux_succes_pct": round(taux, 1),
                "canaux": resultats,
                "verdict": verdict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
