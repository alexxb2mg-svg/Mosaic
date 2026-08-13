"""Le nettoyage déterministe de requête rapporte-t-il ? Mesure sur Alloprof.

Contexte (13/08) : rapporté à BM25, Mosaic atteint 67 % sur Alloprof (questions
d'élèves bruitées) et 86 % sur SciFact (affirmations propres). Ce n'est pas la
langue qui pénalise le moteur, c'est le BRUIT. `mosaic.requete.nettoyer` retire ce
bruit par classes fermées, sans modèle. Diagnostic préalable : 77,3 % des requêtes
Alloprof sont touchées, 13,2 % de caractères retirés en moyenne.

Un SEUL index est construit ; les deux bras ne diffèrent que par la requête posée
— toute différence vient donc du prétraitement, et de rien d'autre.

CRITÈRE D'ADOPTION DÉCLARÉ AVANT LA MESURE : +3 points de rappel sur Alloprof,
sans dégradation sur SciFact. En dessous : documenté et enterré, comme l'expansion
de postings (+0,83 pt) avant lui.

Usage : python research/pretraitement_requete.py <corpus> <verite.jsonl>
            [--top 10] [--temoin]   (--temoin : critère = non-dégradation)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.index import Index
from mosaic.requete import nettoyer


def evaluer(idx: Index, requetes: list[dict], transformer, k: int) -> dict:
    rappel = mrr = 0.0
    t0 = time.perf_counter()
    for q in requetes:
        texte = transformer(q["query"])
        ids = [h["id"] for h in idx.search(texte, k=k)]
        rel = set(q["relevant"])
        touches = [i for i, d in enumerate(ids) if d in rel]
        if touches:
            mrr += 1.0 / (touches[0] + 1)
        rappel += len(set(ids) & rel) / len(rel)
    n = max(1, len(requetes))
    return {
        f"rappel@{k}": round(rappel / n, 4),
        "mrr": round(mrr / n, 4),
        "duree_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: python research/pretraitement_requete.py <corpus> <verite.jsonl> [--top 10]"
        )
        return 2
    corpus, verite = Path(sys.argv[1]), Path(sys.argv[2])
    k = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    requetes = [
        json.loads(li)
        for li in verite.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        idx = Index.build(corpus, Path(tmp) / "idx", index_paths=False)
        print(
            f"index construit en {time.perf_counter() - t0:.0f}s "
            f"({len(idx.ids)} docs, {len(requetes)} requêtes)\n"
        )

        brut = evaluer(idx, requetes, lambda q: q, k)
        net = evaluer(idx, requetes, lambda q: nettoyer(q)[0], k)

    ecart_r = 100 * (net[f"rappel@{k}"] - brut[f"rappel@{k}"])
    ecart_m = 100 * (net["mrr"] - brut["mrr"])
    print(f"{'bras':12s} {'rappel':>9s} {'mrr':>9s}")
    print(f"{'brut':12s} {brut[f'rappel@{k}']:9.4f} {brut['mrr']:9.4f}")
    print(f"{'nettoyé':12s} {net[f'rappel@{k}']:9.4f} {net['mrr']:9.4f}")
    print(f"{'écart (pts)':12s} {ecart_r:+9.2f} {ecart_m:+9.2f}")
    # Deux rôles de corpus, deux critères DIFFÉRENTS. Appliquer le critère de gain à
    # un corpus TÉMOIN affichait « à enterrer » sur un +0.00 parfait (vécu sur
    # SciFact) : le verdict mentait alors que le chiffre était exactement le bon.
    if "--temoin" in sys.argv:
        verdict = (
            "OK — aucune dégradation" if ecart_r >= -0.5 else "DÉGRADE le corpus témoin"
        )
        print(f"\nCorpus TÉMOIN, critère = non-dégradation -> {verdict}")
    else:
        verdict = "ADOPTER" if ecart_r >= 3.0 else "SOUS LE SEUIL DÉCLARÉ (+3 pts)"
        print(f"\nCorpus CIBLE, critère = +3 pts de rappel -> {verdict}")

    Path(__file__).with_name(
        f"resultats_pretraitement_{corpus.parent.name}.json"
    ).write_text(
        json.dumps(
            {
                "brut": brut,
                "nettoye": net,
                "ecart_rappel_pts": round(ecart_r, 2),
                "ecart_mrr_pts": round(ecart_m, 2),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
