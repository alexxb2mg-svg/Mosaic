"""Piste E4 — l'ORDRE des mots apporte-t-il quelque chose que le sac de mots ignore ?

Mosaic est un sac de mots : `encode()` superpose les vecteurs des tokens, l'ordre
est perdu (seules les collocations en gardent une trace). La lignée APNN
(Rachkovskij, arXiv 2112.15475) montre qu'on peut encoder l'ordre DANS le même
vecteur avec une similarité GRADUÉE — deux occurrences à distance d se recouvrent
proportionnellement à (R−d)/R, R étant le « rayon de similarité ».

ADAPTATION DÉCLARÉE (ce n'est PAS la construction exacte du papier) : le papier
encode des positions ABSOLUES et compare des séquences entières (mots d'un
correcteur orthographique, protéines). Un document de 500 mots contre une requête
de 5 mots n'a pas de position absolue commune — on encode donc l'ordre LOCAL :
pour chaque paire de tokens à distance d ≤ R, le couple (t_i, t_{i+d}) contribue
au vecteur avec le poids (R−d+1)/R et la permutation de rang d. La similarité
reste graduée par la distance, et « A avant B » ne vaut pas « B avant A ».

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 17h45) :

  P-E4a — le canal d'ordre SEUL est nettement plus faible que la grille (il ne
          voit que des couples, pas le sujet) : moins de la moitié de son rappel.

  P-E4b — il RATTRAPE des requêtes que la grille rate. C'est la seule question qui
          compte : le canal grammatical est mort de ne rien rattraper (14/08).
          Seuil déclaré, le même que pour lui : **≥ 1 % des requêtes rattrapées**
          (≥ 24 sur Alloprof) pour justifier une place en fusion.

  P-E4c — l'apport, s'il existe, viendra des requêtes CONTENANT UNE EXPRESSION
          FIGÉE (« centre de gravité », « nombre premier ») plutôt que des
          questions longues, où l'ordre est trop variable pour aider.

CRITÈRE D'ADOPTION, déclaré avant : P-E4b tenue (≥ 24 rattrapages nets, c'est-à-
dire rattrapées MOINS perdues). Sinon : documenté et écarté comme le grammatical.

RÉSULTAT PRÉLIMINAIRE (14/08, R=2, **100 requêtes sur 2 316** — échantillon
déclaré, banc complet NON payé faute d'intérêt vu le signal) :

    fusion 0,5550 rappel / 0,3852 MRR
    ordre seul 0,1800 / 0,1064          (P-E4a TENUE : un tiers de la grille)
    combiné 0,5167 / 0,3135             (le canal COÛTE 4 pts de rappel)
    rattrapées 5, perdues 6 -> net -1   (P-E4b FAUSSE, seuil +1)

**ÉCARTÉ.** Même mur que le canal grammatical, mesuré le même jour : il ne
rattrape pas ce que la grille rate. Deux tentatives indépendantes d'ajouter la
structure COMME CANAL SÉPARÉ, deux échecs de même nature — l'hypothèse qui reste
debout est que, dans des questions en français libre, l'ordre des mots ne porte
pas d'information discriminante que le sac de mots aurait perdue. Ce qui n'est PAS
réfuté : encoder l'ordre DANS le vecteur de contenu (la construction exacte du
papier, positions absolues + équivariance) sur des séquences COURTES — c'est le
terrain où Rachkovskij mesure ses gains (correction orthographique, protéines),
et ce n'est pas le nôtre.

Usage : python research/canal_ordre.py [--rayons 1,2,3] [--echantillon 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.docio import tokenize  # noqa: E402
from mosaic.index import Index  # noqa: E402
from mosaic.lexicon import canonicalize, compile_lexicon, load_lexicon  # noqa: E402

# Chemins relatifs au dépôt, surchargeables : ce banc doit être rejouable ailleurs.
CORPUS = Path(os.environ.get("MOSAIC_BENCH_CORPUS", RACINE / "bench/alloprof/corpus"))
VERITE = Path(
    os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/alloprof/verite.jsonl")
)
POTION = Path(
    os.environ.get("MOSAIC_POTION", RACINE / "data_externes/potion_fr_abtt2.msee")
)
DIM = 1024  # dimension du canal d'ordre — banc de recherche, pas la géométrie de prod
K_RRF = 60


def vecteur_token(token: str, rng_cache: dict[str, np.ndarray]) -> np.ndarray:
    """Vecteur pseudo-aléatoire DÉTERMINISTE d'un token (graine = hash du token).
    Déterminisme obligatoire : deux exécutions doivent rendre le même index."""
    v = rng_cache.get(token)
    if v is None:
        graine = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
        v = np.random.default_rng(graine).standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rng_cache[token] = v
    return v


def encoder_ordre(
    tokens: list[str], rayon: int, cache: dict[str, np.ndarray]
) -> np.ndarray:
    """Somme pondérée des COUPLES (t_i, t_{i+d}) pour d ≤ rayon.

    Le couple est encodé par produit terme à terme du premier vecteur avec le
    second DÉCALÉ de d (roll) : l'opération n'est pas commutative, donc « A avant
    B » ≠ « B avant A ». Poids (rayon−d+1)/rayon : les mots voisins comptent plus
    que les mots éloignés — c'est la similarité graduée du papier."""
    if len(tokens) < 2:
        return np.zeros(DIM, dtype=np.float32)
    mat = np.stack([vecteur_token(t, cache) for t in tokens])
    v = np.zeros(DIM, dtype=np.float32)
    for d in range(1, rayon + 1):
        if len(tokens) <= d:
            break
        poids = np.float32((rayon - d + 1) / rayon)
        v += poids * (mat[:-d] * np.roll(mat[d:], d, axis=1)).sum(axis=0)
    n = float(np.linalg.norm(v))
    return v / np.float32(n) if n > 0 else v


def rangs(scores: np.ndarray) -> np.ndarray:
    """Rang 1-based de chaque document (argsort stable, déterministe)."""
    ordre = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int32)
    r[ordre] = np.arange(1, len(scores) + 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rayons", default="1,2,3")
    ap.add_argument("--echantillon", type=int, default=0, help="0 = toutes")
    args = ap.parse_args()
    rayons = [int(x) for x in args.rayons.split(",")]

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

        # Le canal d'ordre voit EXACTEMENT le même flux de tokens que la grille
        # (canonicalisé) — sinon on comparerait deux mondes différents.
        compiled = compile_lexicon(load_lexicon())
        textes = {
            p.name: p.read_text(encoding="utf-8", errors="replace")
            for p in CORPUS.iterdir()
        }
        toks_doc = [canonicalize(tokenize(textes[i]), compiled) for i in idx.ids]

        resultats: dict[int, dict] = {}
        for rayon in rayons:
            t0 = time.perf_counter()
            cache: dict[str, np.ndarray] = {}
            mat_ordre = np.stack([encoder_ordre(t, rayon, cache) for t in toks_doc])
            print(
                f"[R={rayon}] canal construit en {time.perf_counter() - t0:.0f}s",
                flush=True,
            )

            ok_fusion: set[str] = set()
            ok_ordre: set[str] = set()
            ok_combine: set[str] = set()
            rap_o = mrr_o = rap_f = mrr_f = rap_c = mrr_c = 0.0
            for q in requetes:
                pertinents = set(q["relevant"])
                hits = idx.search(q["query"], k=10, fusion=True)
                ids_f = [h["id"] for h in hits]
                if set(ids_f) & pertinents:
                    ok_fusion.add(q["query"])
                    rap_f += len(set(ids_f) & pertinents) / len(pertinents)
                    mrr_f += 1 / next(
                        i for i, d in enumerate(ids_f, 1) if d in pertinents
                    )

                qo = encoder_ordre(
                    canonicalize(tokenize(q["query"]), compiled), rayon, cache
                )
                sc = mat_ordre @ qo if qo.any() else np.zeros(len(idx.ids))
                r_ordre = rangs(sc)
                top_o = [idx.ids[j] for j in np.argsort(-sc, kind="stable")[:10]]
                if set(top_o) & pertinents:
                    ok_ordre.add(q["query"])
                    rap_o += len(set(top_o) & pertinents) / len(pertinents)
                    mrr_o += 1 / next(
                        i for i, d in enumerate(top_o, 1) if d in pertinents
                    )

                # Fusion RRF des rangs de la fusion existante + du canal d'ordre
                rrf: dict[str, float] = {}
                for rang, doc in enumerate(ids_f, 1):
                    rrf[doc] = rrf.get(doc, 0.0) + 1 / (K_RRF + rang)
                for j, doc in enumerate(idx.ids):
                    if sc[j] != 0:
                        rrf[doc] = rrf.get(doc, 0.0) + 1 / (K_RRF + int(r_ordre[j]))
                top_c = [d for d, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:10]]
                if set(top_c) & pertinents:
                    ok_combine.add(q["query"])
                    rap_c += len(set(top_c) & pertinents) / len(pertinents)
                    mrr_c += 1 / next(
                        i for i, d in enumerate(top_c, 1) if d in pertinents
                    )

            n = len(requetes)
            rattrapees = ok_ordre - ok_fusion
            perdues = ok_fusion - ok_combine
            resultats[rayon] = {
                "fusion": {"rappel": round(rap_f / n, 4), "mrr": round(mrr_f / n, 4)},
                "ordre_seul": {
                    "rappel": round(rap_o / n, 4),
                    "mrr": round(mrr_o / n, 4),
                },
                "combine": {"rappel": round(rap_c / n, 4), "mrr": round(mrr_c / n, 4)},
                "rattrapees": len(rattrapees),
                "perdues": len(perdues),
                "net": len(rattrapees) - len(perdues),
                "exemples_rattrapees": sorted(rattrapees)[:5],
            }
            r = resultats[rayon]
            print(
                f"[R={rayon}] fusion {r['fusion']['rappel']:.4f}/{r['fusion']['mrr']:.4f}"
                f" · ordre seul {r['ordre_seul']['rappel']:.4f}/{r['ordre_seul']['mrr']:.4f}"
                f" · combiné {r['combine']['rappel']:.4f}/{r['combine']['mrr']:.4f}"
                f" · rattrapées {r['rattrapees']} / perdues {r['perdues']}"
                f" (net {r['net']:+d})",
                flush=True,
            )

    meilleur = max(resultats.values(), key=lambda x: x["net"])
    seuil = max(1, len(requetes) // 100)
    print(
        f"\nP-E4b : net max {meilleur['net']:+d} contre un seuil de {seuil} -> "
        + ("TENUE" if meilleur["net"] >= seuil else "FAUSSE")
    )
    verdict = "ADOPTÉ en fusion" if meilleur["net"] >= seuil else "ÉCARTÉ"
    print(f"CRITÈRE -> {verdict}")
    Path(__file__).with_name("resultats_canal_ordre.json").write_text(
        json.dumps(
            {
                "requetes": len(requetes),
                "dim": DIM,
                "resultats": resultats,
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
