"""Méta-recherche cross-index : une question, plusieurs corpus (devis, comptabilité, projets,
courriels, notes), une liste fusionnée.

Le piège : les scores de deux index différents ne sont PAS comparables (échelles distinctes selon
le vocabulaire du corpus). Concaténer par score brut laisserait l'index à la plus grande échelle
tout rafler. La fusion par RANG (Reciprocal Rank Fusion, Cormack et al. 2009) est invariante à
l'échelle : chaque résultat compte pour 1/(k + rang_local), on somme, on retrie. C'est la monnaie
juste entre corpus dont on ne peut pas comparer les scores.

Ce module ne TRANCHE pas la pertinence — il assure le RAPPEL (faire remonter les candidats de tous
les corpus) et conserve la PROVENANCE et le score local de chaque résultat, pour que les critères
de pertinence (docs/criteres_pertinence.md) s'appliquent ensuite, par un agent ou un humain. Un
résumé par index signale quand un corpus paraît hors-sujet (son meilleur score reste très bas).
"""

from collections.abc import Iterable

K_RRF_DEFAULT = (
    60  # constante RRF standard : amortit l'écart entre les tout premiers rangs
)


def rrf_fuse(
    listes: Iterable[tuple[str, list[dict]]],
    k: int = 10,
    k_rrf: int = K_RRF_DEFAULT,
) -> list[dict]:
    """Fusionne des listes classées venant de sources distinctes par Reciprocal Rank Fusion.

    `listes` : itérable de (nom_source, résultats), chaque résultat un dict portant au moins `id`
    (et idéalement `score`). Retourne les `k` meilleurs, chacun enrichi de :
      - `index` : la source d'où il vient (provenance) ;
      - `rang_local` : son rang 1-based dans sa source ;
      - `score_local` : son score brut dans sa source (non comparable entre sources) ;
      - `score_rrf` : la contribution RRF fusionnée (base du classement final).

    Un même `id` présent dans PLUSIEURS sources voit ses contributions RRF s'additionner (co-
    signalé par plusieurs corpus = remonté). Entre corpus disjoints, RRF entrelace par rang — le
    comportement correct quand les scores ne sont pas comparables."""
    if k < 1:
        raise ValueError("k doit être >= 1")
    if k_rrf < 1:
        raise ValueError("k_rrf doit être >= 1")
    fusion: dict[tuple[str, str], dict] = {}
    for source, resultats in listes:
        for rang, r in enumerate(resultats, start=1):
            doc_id = r["id"]
            cle = (source, doc_id)
            contrib = 1.0 / (k_rrf + rang)
            if cle in fusion:
                fusion[cle]["score_rrf"] += contrib
            else:
                fusion[cle] = {
                    "index": source,
                    "id": doc_id,
                    "rang_local": rang,
                    "score_local": round(float(r.get("score", 0.0)), 4),
                    "score_rrf": contrib,
                }
    ordre = sorted(
        fusion.values(),
        key=lambda x: (-x["score_rrf"], x["index"], x["rang_local"]),
    )
    for x in ordre:
        x["score_rrf"] = round(x["score_rrf"], 6)
    return ordre[:k]


def znorm_fuse(listes: Iterable[tuple[str, list[dict]]], k: int = 10) -> list[dict]:
    """Fusionne par scores Z-NORMALISÉS PAR SOURCE — pour des sources HOMOGÈNES.

    Mesuré (`research/shards_fusion.py`, Alloprof découpé en 4 shards équilibrés,
    2 316 requêtes) : **0.3832 R@10 / 0.2229 MRR, contre 0.3216 / 0.2164 pour
    l'index unique** — la fusion bat l'index entier sur les DEUX métriques, là où
    le RRF gagnait le rappel (0.3809) en payant la précision de tête (0.1713).

    Lecture du gain : chaque shard apprend ses propres profils de cooccurrence, ses
    erreurs se décorrèlent de celles des autres (même mécanisme que la fusion
    multi-canaux), et la z-normalisation rend leurs échelles comparables sans
    écraser l'écart entre le premier et le second — ce que le rang, lui, écrase.

    QUAND L'UTILISER — et c'est la condition qui décide : des sources de MÊME
    NATURE (un corpus découpé en tranches, idéalement de tailles voisines). Sur des
    corpus HÉTÉROGÈNES (devis vs comptabilité vs chantiers), la z-normalisation
    n'a pas de sens : deux distributions de scores portant sur des mondes différents
    ne deviennent pas comparables parce qu'on les centre. C'est le RRF qui reste le
    défaut de `mosaic meta`, et il n'est pas question de changer ça sans l'avoir
    mesuré sur des corpus réellement hétérogènes.

    Biais connu et non corrigé : sur des tranches de tailles TRÈS inégales, la
    petite source reste sur-représentée (3.81x sa part mesurée sur un découpage
    6/19/25/50 %). Pondérer les contributions par la part de corpus a été tenté et
    ÉCHOUE (elle écrase totalement les petites sources, R@10 0.2051) — la correction
    utile agirait sur la PROFONDEUR demandée à chaque source, pas sur le poids de
    ses scores. D'où la règle de conception : viser des tranches équilibrées.
    """
    if k < 1:
        raise ValueError("k doit être >= 1")
    fusion: list[dict] = []
    for source, resultats in listes:
        scores = [float(r.get("score", 0.0)) for r in resultats]
        if not scores:
            continue
        mu = sum(scores) / len(scores)
        var = sum((s - mu) ** 2 for s in scores) / len(scores)
        sigma = var**0.5 or 1.0  # source à scores identiques : pas d'échelle à corriger
        for rang, (r, s) in enumerate(zip(resultats, scores, strict=True), start=1):
            fusion.append(
                {
                    "index": source,
                    "id": r["id"],
                    "rang_local": rang,
                    "score_local": round(s, 4),
                    "score_z": round((s - mu) / sigma, 6),
                }
            )
    fusion.sort(key=lambda x: (-x["score_z"], x["index"], x["rang_local"]))
    return fusion[:k]


def resume_par_index(listes: Iterable[tuple[str, list[dict]]]) -> list[dict]:
    """Diagnostic de RAPPEL : par source, le nombre de candidats et le meilleur score local.
    Un corpus dont le meilleur score reste très bas est probablement hors-sujet pour la question —
    signalé, jamais écarté d'office (cf. principe « cas ambigu → ne pas deviner »)."""
    out = []
    for source, resultats in listes:
        meilleur = max((float(r.get("score", 0.0)) for r in resultats), default=0.0)
        out.append(
            {
                "index": source,
                "candidats": len(resultats),
                "meilleur_score_local": round(meilleur, 4),
            }
        )
    return out
