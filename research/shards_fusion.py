"""Biais de fusion inter-shards — le juge du sharding (piste du 12/08 au soir).

QUESTION : découper un corpus en N index (shards) et fusionner les résultats
par requête permet-il de contourner le plafond de taille SANS perdre de
qualité ? Où est le biais quand les shards sont de tailles inégales ?

PROTOCOLE : Alloprof (2 556 docs, 2 316 requêtes réelles, vérité terrain
publique), moteur réel (Index.build/search, défauts + index_paths=False,
même config partout — seule l'ARCHITECTURE varie) :

  - baseline : index unique ;
  - découpage ÉQUILIBRÉ : 4 shards ~639 docs (sha256(doc_id) % 4 —
    déterministe et décorrélé du contenu, jamais alphabétique) ;
  - découpage INÉGAL : 4 shards ~6 % / ~19 % / ~25 % / ~50 % (tranches de
    sha256 % 16) — le cas qui expose le biais de rang ;
  - fusions comparées sur les top-10 locaux de chaque shard :
      (a) SCORE BRUT : concaténation triée par cosinus (les scores d'une même
          config sont-ils comparables entre shards, malgré des IDF/profils
          appris par shard ?) ;
      (b) RRF K=60 sur les rangs locaux (ce que fait `mosaic meta`).

PRÉDICTIONS DÉCLARÉES AVANT MESURE :
  P1 — le score brut inter-shards reste PROCHE de l'index unique (dérive de
       quelques points au plus : les profils/IDF divergent par shard mais la
       géométrie des grilles est partagée via les signatures SHA) ;
  P2 — le RRF inter-shards fait PIRE que le score brut sur le découpage
       INÉGAL : le rang 3 d'un shard de 160 docs ne vaut pas le rang 3 d'un
       shard de 1 280 docs — sur-représentation du petit shard attendue ;
  P3 — le découpage équilibré souffre moins que l'inégal (les deux fusions).

MESURES : R@10 et MRR@10 sur les 2 316 requêtes ; et pour chaque shard du
découpage inégal, sa part des top-10 fusionnés vs sa part du corpus (le
ratio > 1 = sur-représentation, la signature directe du biais).

VERDICTS PASSE 1 (mesurés, prédictions confrontées) :
  P1 FALSIFIÉE — le score brut inter-shards perd ~5,3 pts de R@10 MÊME à
     tailles égales : les IDF/profils appris par shard décalent les échelles
     de cosinus, la comparaison directe est structurellement fragile ;
  P2 FALSIFIÉE en rappel — le RRF équilibré BAT l'index unique en R@10
     (0.3809 vs 0.3216, +5,9 pts !) en payant le MRR (−4,5 pts) : profil
     « canal de rappel », chaque shard fait remonter SES meilleurs candidats ;
     mais le biais prédit existe : sur l'inégal, le shard de 6 % truste 4,0×
     sa part des top-10 (le rang k d'un petit index vaut moins, RRF l'ignore) ;
  P3 CONFIRMÉE — l'équilibré souffre moins (RRF 0.3809 vs 0.3391).

PASSE 2 — corrections du biais, prédictions déclarées avant mesure :
  P4 — z-normalisation des scores PAR SHARD (μ/σ des k scores locaux) :
       rétablit la comparabilité que le brut n'a pas — attendu ≥ brut + équité ;
  P5 — RRF PONDÉRÉ par la part de corpus du shard : ramène la
       sur-représentation 4,0 vers ~1 et remonte le R@10 de l'inégal vers
       celui de l'équilibré.

VERDICTS PASSE 2 (mesurés) :
  P4 CONFIRMÉE AU-DELÀ DE L'ATTENDU — équilibré+znorm : R@10 0.3832 /
     MRR 0.2229, BAT L'INDEX UNIQUE SUR LES DEUX MÉTRIQUES (0.3216/0.2164).
     Lecture : effet d'ensemble — chaque shard apprend SES profils, les
     erreurs des lecteurs se décorrèlent (même mécanisme que la fusion
     multi-canaux), et la z-norm rend leurs échelles comparables sans
     sacrifier la précision de tête comme le fait le RRF. Caveat honnête :
     μ/σ estimés sur k=10 scores seulement — fragile en théorie, gagnant ici ;
     sur l'inégal, le biais du petit shard persiste (3.81×) mais coûte moins
     (0.3488/0.2056, meilleure fusion du découpage inégal).
  P5 FALSIFIÉE — désastre : pondérer les contributions RRF par la part de
     corpus ÉCRASE les petits shards (0 doc au top-10, R@10 0.2051 pire que
     tout) — la correction naïve sur-corrige totalement. Une pondération
     utile devrait agir sur la PROFONDEUR demandée au shard, pas sur le
     poids de ses rangs.

RÈGLE DE CONCEPTION ÉTABLIE : sharder à TAILLES ÉGALES (plafond de volume
par index, ouverture du suivant une fois plein) + fusion par z-normalisation
des scores par shard. La recherche shardée coûte +13 % en séquentiel et se
parallélise par shard ; le build, lui, est borné par le plus gros shard.

Usage : python research/shards_fusion.py <corpus_alloprof> <verite.jsonl> <dossier_travail>
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.index import Index

K_RRF = 60
TOP = 10


def _hash16(doc_id: str) -> int:
    return hashlib.sha256(doc_id.encode("utf-8")).digest()[0] % 16


def _shard_equilibre(doc_id: str) -> int:
    return _hash16(doc_id) % 4


def _shard_inegal(doc_id: str) -> int:
    h = _hash16(doc_id)
    if h == 0:
        return 0  # ~6 %
    if h <= 3:
        return 1  # ~19 %
    if h <= 7:
        return 2  # ~25 %
    return 3  # ~50 %


def _construire(corpus: Path, travail: Path, nom: str, routeur) -> list[Index]:
    """Copie chaque document dans le dossier de son shard puis construit les 4 index."""
    racine = travail / nom
    if not (racine / "pret.txt").is_file():
        for p in sorted(corpus.glob("*.md")):
            cible = racine / f"corpus_{routeur(p.name)}" / p.name
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, cible)
        for s in range(4):
            Index.build(racine / f"corpus_{s}", racine / f"idx_{s}", index_paths=False)
        (racine / "pret.txt").write_text("ok", encoding="utf-8")
    return [Index.open(racine / f"idx_{s}") for s in range(4)]


def _fusion_score_brut(par_shard: list[list[dict]]) -> list[str]:
    tous = [h for res in par_shard for h in res]
    tous.sort(key=lambda h: (-h["score"], h["id"]))
    return [h["id"] for h in tous[:TOP]]


def _fusion_rrf(par_shard: list[list[dict]]) -> list[str]:
    scores: dict[str, float] = {}
    for res in par_shard:
        for rang, h in enumerate(res):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (K_RRF + rang + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:TOP]


def _fusion_znorm(par_shard: list[list[dict]]) -> list[str]:
    """Scores z-normalisés PAR SHARD (μ/σ de ses k scores locaux) puis concaténés.
    Rend les échelles comparables sans toucher à l'ordre local de chaque shard."""
    tous: list[tuple[float, str]] = []
    for res in par_shard:
        if not res:
            continue
        vals = [h["score"] for h in res]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = var**0.5 or 1.0
        tous.extend(((h["score"] - mu) / sigma, h["id"]) for h in res)
    tous.sort(key=lambda t: (-t[0], t[1]))
    return [d for _, d in tous[:TOP]]


def _fusion_rrf_pondere(par_shard: list[list[dict]], tailles: list[int]) -> list[str]:
    """RRF dont chaque contribution est pondérée par la part de corpus du shard —
    le rang k d'un petit index ne vaut plus autant que celui d'un grand."""
    total = sum(tailles)
    scores: dict[str, float] = {}
    for res, taille in zip(par_shard, tailles, strict=True):
        poids = taille / total
        for rang, h in enumerate(res):
            scores[h["id"]] = scores.get(h["id"], 0.0) + poids / (K_RRF + rang + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:TOP]


def _metriques(tops: list[list[str]], pertinents: list[set[str]]) -> dict:
    r10 = mrr = 0.0
    for top, rel in zip(tops, pertinents, strict=True):
        touches = [i for i, d in enumerate(top) if d in rel]
        if touches:
            mrr += 1.0 / (touches[0] + 1)
        r10 += len(set(top) & rel) / len(rel)
    n = len(tops)
    return {"R@10": round(r10 / n, 4), "MRR@10": round(mrr / n, 4)}


def main() -> int:
    corpus, verite_p, travail = (Path(a) for a in sys.argv[1:4])
    requetes = [
        json.loads(li) for li in verite_p.read_text(encoding="utf-8").splitlines() if li
    ]
    pertinents = [set(q["relevant"]) for q in requetes]

    # --- baseline : index unique --------------------------------------------------
    idx_unique_dir = travail / "idx_unique"
    if not idx_unique_dir.is_dir():
        Index.build(corpus, idx_unique_dir, index_paths=False)
    unique = Index.open(idx_unique_dir)
    t0 = time.perf_counter()
    tops_unique = [
        [h["id"] for h in unique.search(q["query"], k=TOP)] for q in requetes
    ]
    t_unique = time.perf_counter() - t0

    resultats = {
        "unique": {**_metriques(tops_unique, pertinents), "s_total": round(t_unique, 1)}
    }

    # --- les deux découpages ------------------------------------------------------
    for nom, routeur in (("equilibre", _shard_equilibre), ("inegal", _shard_inegal)):
        shards = _construire(corpus, travail, nom, routeur)
        tailles = [s.stats()["docs"] for s in shards]
        t0 = time.perf_counter()
        locaux = [[s.search(q["query"], k=TOP) for s in shards] for q in requetes]
        t_shards = time.perf_counter() - t0
        tops_brut = [_fusion_score_brut(loc) for loc in locaux]
        tops_rrf = [_fusion_rrf(loc) for loc in locaux]
        tops_znorm = [_fusion_znorm(loc) for loc in locaux]
        tops_rrfp = [_fusion_rrf_pondere(loc, tailles) for loc in locaux]
        bloc = {
            "tailles": tailles,
            "score_brut": _metriques(tops_brut, pertinents),
            "rrf": _metriques(tops_rrf, pertinents),
            "znorm": _metriques(tops_znorm, pertinents),
            "rrf_pondere": _metriques(tops_rrfp, pertinents),
            "s_total_recherche": round(t_shards, 1),
        }
        # signature directe du biais : part des top-10 fusionnés par shard vs part corpus
        appartenance = {d: i for i, s in enumerate(shards) for d in s.ids}
        total_docs = sum(tailles)
        for cle, tops in (
            ("brut", tops_brut),
            ("rrf", tops_rrf),
            ("znorm", tops_znorm),
            ("rrf_pondere", tops_rrfp),
        ):
            comptes = [0] * 4
            for top in tops:
                for d in top:
                    comptes[appartenance[d]] += 1
            total = sum(comptes)
            bloc[f"surrepresentation_{cle}"] = [
                round((c / total) / (tailles[i] / total_docs), 2)
                for i, c in enumerate(comptes)
            ]
        resultats[nom] = bloc

    print(json.dumps(resultats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
