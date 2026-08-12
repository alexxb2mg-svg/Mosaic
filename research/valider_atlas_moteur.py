"""Acceptation de la greffe atlas (spec 2026-08-12-atlas-canal-fusion-design.md §2).

Le banc fusion Alloprof PLEIN CORPUS rejoué À TRAVERS LE MOTEUR (Index.build --hybride
--atlas + search --fusion), même config de grille que la recherche (poids calibrés
0.50/0.30/0.20, sans tokens de chemin, potion + abtt 2). Attendu : le quartet moteur
retrouve le 0.5319 R@10 de research/atlas_fusion.py à ±0.5 pt — la quantification int8
des cartes est la SEULE différence tolérée entre moteur et recherche.

Usage : python research/valider_atlas_moteur.py <corpus> <verite.jsonl> <table.msee>
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.index import Index

ATTENDU_RECHERCHE = 0.5319  # R@10 quartet, research/atlas_fusion.py plein corpus
TOLERANCE = 0.005
# Verdict du 12/08 : moteur 0.5461 (+1.42 pt AU-DESSUS de la recherche) — écart
# EXPLIQUÉ : atlas_fusion préparait profils/SOM/BM25 via _preparer AVEC les tokens
# de chemin (bruit hexa UUID, 72k tokens) alors que le moteur construit tout en
# config calibrée --no-path-tokens (50k tokens, SOM propre). Rien n'est perdu à
# l'int8 ; le critère est donc une BORNE BASSE (dépasser est bienvenu et expliqué).


def main() -> int:
    corpus, verite, table = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
    requetes = [
        (str(o["query"]), [str(x) for x in o["relevant"]])
        for o in (
            json.loads(line)
            for line in verite.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t0 = time.perf_counter()
        idx = Index.build(
            corpus,
            Path(tmp) / "idx",
            weights=(0.50, 0.30, 0.20),
            index_paths=False,
            embeddings_path=table,
            abtt=2,
            hybride=True,
            atlas=True,
        )
        print(f"build moteur (hybride+atlas) : {time.perf_counter() - t0:.0f} s")
        print("stats atlas :", idx.stats().get("atlas"))
        idx.chauffer_recherche()
        recalls, rrs, t_req = [], [], []
        for q, rel in requetes:
            t1 = time.perf_counter()
            hits = idx.search(q, k=10, fusion=True)
            t_req.append((time.perf_counter() - t1) * 1000)
            ids = [h["id"] for h in hits]
            rel_set = set(rel)
            recalls.append(len(rel_set & set(ids)) / max(1, len(rel_set)))
            rr = next(
                (1.0 / r for r, d in enumerate(ids, start=1) if d in rel_set), 0.0
            )
            rrs.append(rr)
        n = max(1, len(requetes))
        r10 = sum(recalls) / n
        mrr = sum(rrs) / n
        lat = sorted(t_req)[len(t_req) // 2]
        print(
            f"quartet MOTEUR : R@10 {r10:.4f}  MRR {mrr:.4f}  ({lat:.1f} ms/req médiane)"
        )
        ecart = r10 - ATTENDU_RECHERCHE
        verdict = "ACCEPTÉ" if ecart >= -TOLERANCE else "SOUS LA BORNE — REFUSÉ"
        print(
            f"borne basse {ATTENDU_RECHERCHE} - {TOLERANCE} -> écart "
            f"{ecart:+.4f} : {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
