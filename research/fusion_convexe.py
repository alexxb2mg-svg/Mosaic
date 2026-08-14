"""Piste F2 — la combinaison convexe bat-elle notre RRF sur nos corpus ?

Nous fusionnons quatre canaux par RRF : chaque canal vote 1/(60+rang), on somme.
C'est robuste et sans réglage — mais cela JETTE L'AMPLITUDE. Un document que la
grille place premier avec un écart énorme sur le second pèse exactement autant
qu'un premier arraché de justesse. Sur quatre canaux dont deux rendent des scores
très plats, c'est un aplatissement de plus.

Bruch, Gai, Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval*
(arXiv 2210.11934, TOIS 2023) mesurent que RRF est sensible à ses paramètres et
que la combinaison convexe normalisée le bat, en domaine comme hors domaine.
Elasticsearch et OpenSearch exposent d'ailleurs tous deux un mode linéaire à côté
de leur RRF.

CE QU'ON MESURE : les deux fusions sur les MÊMES scores de canaux, donc seule la
fonction de combinaison diffère. Trois variantes de normalisation (min-max,
z-score, et rang comme témoin = RRF), et pour la combinaison convexe deux
réglages de poids : uniforme (aucun ajustement) et calibré sur une moitié des
requêtes puis évalué sur l'autre (pour ne pas mesurer un sur-apprentissage).

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 21h55) :

  P-F2a — la combinaison convexe à poids UNIFORMES ne bat pas RRF : sans
          calibration, normaliser des échelles hétérogènes ne suffit pas.
  P-F2b — la combinaison convexe CALIBRÉE bat RRF d'au moins 2 points de rappel
          sur la moitié d'évaluation. C'est le chiffre qui décide.
  P-F2c — l'écart entre min-max et z-score sera négligeable (le papier le dit :
          le choix de normalisation est secondaire).

RÉSERVE DÉCLARÉE : le papier n'étudie que DEUX canaux ; à quatre, il y a trois
poids libres, donc plus de risque de sur-apprendre. D'où la séparation
calibration/évaluation, sans laquelle le résultat ne voudrait rien dire.

CRITÈRE D'ADOPTION : P-F2b tenue sur la moitié d'ÉVALUATION (jamais sur celle de
calibration). Sinon : RRF reste, et l'affaire est documentée.

RÉSULTAT (14/08, 2 316 requêtes Alloprof, `resultats_fusion_convexe.json`) :

    fusion                       rappel@10     MRR
    RRF (production)               0,5206    0,3299
    CC min-max uniforme            0,5281    0,3589
    CC min-max calibrée            0,5425    0,3701   (+2,18 pts)
    CC z-score uniforme            0,5259    0,3530
    CC z-score calibrée            0,5471    0,3733   (+2,65 pts)

**P-F2b TENUE — critère atteint.** Mais deux surprises comptent plus que le
verdict :

1. **P-F2a est FAUSSE : les poids UNIFORMES battent déjà RRF** (+0,75 pt de
   rappel, +2,9 pts de MRR sans le moindre réglage). Ce n'est donc pas la
   calibration qui paie l'essentiel, c'est la NORMALISATION — autrement dit,
   l'amplitude que RRF jette portait vraiment de l'information. La calibration
   n'ajoute que la moitié du gain.

2. **La calibration met le canal GRILLE à 0,0 (z-score) ou 0,5 (min-max)** — le
   canal sémantique propre à Mosaic, celui qui fait sa signature. Le résultat est
   à prendre avec précaution, pour trois raisons : les deux normalisations ne
   s'accordent pas sur la valeur (0,0 contre 0,5), ce qui signale une zone plate
   plutôt qu'un zéro franc ; l'atlas est DÉRIVÉ de la grille, donc une partie de
   son information y survit ; et Alloprof est un corpus scolaire, pas notre
   terrain métier. **À vérifier sur le corpus interne avant toute conclusion** —
   un poids nul sur Alloprof ne dit rien de ce que la grille fait sur des devis.

CONTRE-RÉSULTAT SUR LE CORPUS MÉTIER (14/08, 370 documents, 40 requêtes) — et
il RENVERSE le verdict :

    fusion                       rappel@10     MRR
    RRF (production)               0,6000    0,3581
    CC calibrée (les deux)         0,5000    0,3656   (**−10,00 pts**)
    poids trouvés : bm25 0,5 — atlas, embed, grille TOUS À ZÉRO

La calibration, avec vingt requêtes seulement, tombe sur une solution
DÉGÉNÉRÉE (un canal unique) qui s'effondre sur les vingt autres. Ce n'est pas
un jugement sur les canaux, c'est du sur-apprentissage caractérisé — la réserve
déclarée avant la mesure (« à quatre canaux, trois poids libres, plus de risque
de sur-apprendre ») se réalise à la lettre.

TROIS CONCLUSIONS, contraires à ce que le seul banc Alloprof laissait croire :

1. **La CC n'est PAS adoptable en production.** Un gain qui ne survit pas au
   changement de corpus n'est pas un gain, c'est un réglage. Elle exige un
   volume de requêtes annotées que notre terrain n'a pas — et le papier, qui
   la dit « sample efficient », étudie DEUX canaux, pas quatre.
2. **Le poids nul de la grille sur Alloprof n'est ni confirmé ni infirmé** :
   ici la calibration met TOUT à zéro sauf un canal, donc elle ne mesure rien
   de fiable. La question reste ouverte.
3. **Ce qui manque est nommé : des requêtes métier avec leur bonne réponse.**
   Quelques centaines suffiraient. C'est exactement ce que le journal des
   recherches produira par simple usage, sans annotation — la limite atteinte
   ce soir est celle que l'usage lèvera.

RRF reste en production, et pour une raison qui n'est plus seulement historique :
il n'a aucun paramètre à sur-apprendre.

Usage : python research/fusion_convexe.py [--echantillon 0]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.index import Index  # noqa: E402
from mosaic.meta import K_RRF_DEFAULT  # noqa: E402

CORPUS = Path(os.environ.get("MOSAIC_BENCH_CORPUS", RACINE / "bench/alloprof/corpus"))
VERITE = Path(
    os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/alloprof/verite.jsonl")
)
POTION = Path(
    os.environ.get("MOSAIC_POTION", RACINE / "data_externes/potion_fr_abtt2.msee")
)


def minmax(s: np.ndarray) -> np.ndarray:
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def zscore(s: np.ndarray) -> np.ndarray:
    m, e = float(s.mean()), float(s.std())
    return (s - m) / e if e > 0 else np.zeros_like(s)


def rangs_rrf(s: np.ndarray) -> np.ndarray:
    """Le témoin : ce que fait RRF — l'amplitude est remplacée par le rang."""
    n = s.size
    ordre = np.argsort(-s, kind="stable")
    out = np.empty(n, dtype=np.float64)
    out[ordre] = 1.0 / (K_RRF_DEFAULT + np.arange(1, n + 1))
    return out


NORMALISATIONS = {"rrf": rangs_rrf, "minmax": minmax, "zscore": zscore}


def rappel_mrr(
    ids: list[str], pertinents: set[str], k: int = 10
) -> tuple[float, float]:
    top = ids[:k]
    rappel = len(set(top) & pertinents) / max(1, len(pertinents))
    mrr = 0.0
    for i, d in enumerate(top, 1):
        if d in pertinents:
            mrr = 1.0 / i
            break
    return rappel, mrr


def evaluer(
    canaux_par_requete: list[dict[str, np.ndarray]],
    pertinents_par_requete: list[set[str]],
    ids: list[str],
    normalisation: str,
    poids: dict[str, float],
) -> tuple[float, float]:
    f = NORMALISATIONS[normalisation]
    rappel = mrr = 0.0
    for canaux, pertinents in zip(canaux_par_requete, pertinents_par_requete):
        total = np.zeros(len(ids), dtype=np.float64)
        for nom, scores in canaux.items():
            total += poids.get(nom, 0.0) * f(scores)
        top = np.argsort(-total, kind="stable")[:10]
        r, m = rappel_mrr([ids[i] for i in top], pertinents)
        rappel += r
        mrr += m
    n = max(1, len(canaux_par_requete))
    return rappel / n, mrr / n


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--echantillon", type=int, default=0)
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
        print(f"index construit en {time.perf_counter() - t0:.0f}s", flush=True)

        from mosaic import atlas as atlas_module
        from mosaic import rerank as rerank_module
        from mosaic.docio import tokenize
        from mosaic.lexicon import canonicalize
        from mosaic.queries import _cos_all, merge

        tous, verites = [], []
        t0 = time.perf_counter()
        for n, q in enumerate(requetes, 1):
            texte = q["query"]
            toks = merge(
                merge(canonicalize(tokenize(texte), idx._compiled), idx.colloc),
                idx.colloc,
            )
            canaux = {
                "grille": _cos_all(idx, texte),
                "bm25": idx.bm25.scores(toks) if idx.bm25 is not None else None,
                "embed": idx.rerank_vecs @ rerank_module.encode_query(texte),
            }
            if (
                idx.atlas_positions is not None
                and idx.atlas_mat is not None
                and idx.atlas_norms is not None
            ):
                carte = atlas_module.carte(
                    toks, idx.profiles.rows, idx.atlas_positions, idx.profiles.idf
                )
                nq = float(np.linalg.norm(carte))
                if nq > 0:
                    denom = np.where(idx.atlas_norms == 0, 1.0, idx.atlas_norms) * nq
                    canaux["atlas"] = (idx.atlas_mat.astype(np.float32) @ carte) / denom
            tous.append({k: v for k, v in canaux.items() if v is not None})
            verites.append(set(q["relevant"]))
            if n % 500 == 0:
                print(f"  {n}/{len(requetes)} scores calculés", flush=True)
        print(f"scores en {time.perf_counter() - t0:.0f}s", flush=True)

    noms = sorted(tous[0])
    milieu = len(tous) // 2
    calib = slice(0, milieu)
    evalu = slice(milieu, len(tous))

    # Référence : RRF à poids uniformes — exactement la fusion en production.
    ref_r, ref_m = evaluer(
        tous[evalu], verites[evalu], idx.ids, "rrf", dict.fromkeys(noms, 1.0)
    )
    print(f"\nRRF (production)      rappel@10 {ref_r:.4f}  mrr {ref_m:.4f}")

    resultats = {"rrf_production": {"rappel": round(ref_r, 4), "mrr": round(ref_m, 4)}}
    gains: list[float] = []
    for norm in ("minmax", "zscore"):
        u_r, u_m = evaluer(
            tous[evalu], verites[evalu], idx.ids, norm, dict.fromkeys(noms, 1.0)
        )
        print(f"CC {norm} uniforme     rappel@10 {u_r:.4f}  mrr {u_m:.4f}")

        # Calibration : grille de poids sur la PREMIÈRE moitié seulement.
        meilleur, meilleur_score = None, -1.0
        for combi in itertools.product((0.0, 0.5, 1.0, 2.0), repeat=len(noms)):
            if sum(combi) == 0:
                continue
            poids = dict(zip(noms, combi))
            r, _ = evaluer(tous[calib], verites[calib], idx.ids, norm, poids)
            if r > meilleur_score:
                meilleur, meilleur_score = poids, r
        c_r, c_m = evaluer(tous[evalu], verites[evalu], idx.ids, norm, meilleur or {})
        print(
            f"CC {norm} calibré      rappel@10 {c_r:.4f}  mrr {c_m:.4f}   "
            f"poids {meilleur}"
        )
        gain = round((c_r - ref_r) * 100, 2)
        gains.append(gain)
        resultats[f"cc_{norm}"] = {
            "uniforme": {"rappel": round(u_r, 4), "mrr": round(u_m, 4)},
            "calibre": {"rappel": round(c_r, 4), "mrr": round(c_m, 4)},
            "poids": meilleur,
            "gain_vs_rrf_pts": gain,
        }

    meilleur_gain = max(gains) if gains else 0.0
    print(
        f"\nP-F2b  meilleur gain calibré {meilleur_gain:+.2f} pts -> "
        + ("TENUE" if meilleur_gain >= 2 else "FAUSSE")
    )
    verdict = "ADOPTÉ" if meilleur_gain >= 2 else "RRF CONSERVÉ"
    print(f"CRITÈRE (+2 pts sur la moitié d'évaluation) -> {verdict}")
    resultats["verdict"] = verdict
    Path(__file__).with_name("resultats_fusion_convexe.json").write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
