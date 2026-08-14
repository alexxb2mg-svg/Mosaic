"""Analyse APPROFONDIE du verdict de la piste A — où le cross-encodeur gagne, et pourquoi.

Le banc de 11 h (rerank_croise.py, 300 requêtes) n'a sauvé que les agrégats :
+20,7 pts de MRR, +10,9 de rappel@10, 134 s/requête. Ces trois nombres disent QUE
le gain existe, pas OÙ il se produit — donc pas comment l'exploiter en différé.

Ce script rejoue les deux bras en conservant le DÉTAIL PAR REQUÊTE, sur un
échantillon plus petit (défaut 60 requêtes, ~2 h 15) tiré de la MÊME graine et du
MÊME tirage que le banc complet : ce sont donc les 60 premières requêtes du banc
de 11 h, pas un tirage concurrent — les agrégats doivent se retrouver, c'est le
contrôle de cohérence.

Ce qu'on cherche, qu'aucun agrégat ne peut dire :
  A. La DISTRIBUTION du gain : gain massif sur quelques requêtes, ou petit gain
     partout ? Décide de la stratégie différée — si 20 % des requêtes portent 80 %
     du gain, il faut savoir LESQUELLES et ne rerangher qu'elles.
  B. Les RÉGRESSIONS : combien de requêtes le rerank DÉGRADE-t-il ? Un gain moyen
     de +20 pts peut cacher 15 % de requêtes cassées — inacceptable en production
     même différée, sans garde-fou.
  C. Le PROFIL des gagnantes vs perdantes : longueur de requête, présence de la
     réponse au-delà du rang 10 en fusion (donc rattrapable), position d'origine.
  D. La FENÊTRE UTILE : à quelle profondeur les documents remontés étaient-ils ?
     Si tout vient des rangs 1-20, la fenêtre 50 coûte 60 % de calcul pour rien —
     et le coût différé s'effondre d'autant.

Usage : python research/rerank_analyse.py [--echantillon 60] [--fenetre 50]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple


class Ligne(NamedTuple):
    """Une requête mesurée dans les deux bras — typée, pour que les agrégats le soient."""

    query: str
    mots: int
    n_pertinents: int
    rang_avant: int | None
    rang_apres: int | None
    rr_avant: float
    rr_apres: float
    top10_avant: int
    top10_apres: int
    latence_s: float

    @property
    def gain(self) -> float:
        return self.rr_apres - self.rr_avant


RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.index import Index  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank_croise import (  # noqa: E402
    CORPUS,
    GRAINE,
    POTION,
    VERITE,
    demarrer_serveur,
    reranger,
)


def rang_premier_pertinent(ids: list[str], pertinents: set[str]) -> int | None:
    """Rang 1-based du premier document pertinent, None s'il n'y en a aucun."""
    for i, d in enumerate(ids, 1):
        if d in pertinents:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--echantillon", type=int, default=60)
    ap.add_argument("--fenetre", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    requetes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    # MÊME tirage que le banc complet (même graine, même population) : on garde le
    # préfixe, donc ces requêtes SONT un sous-ensemble de celles déjà mesurées.
    echantillon = random.Random(GRAINE).sample(requetes, 300)[: args.echantillon]
    textes = {
        p.name: p.read_text(encoding="utf-8", errors="replace")
        for p in CORPUS.iterdir()
    }

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
        )
        print(f"index construit en {time.perf_counter() - t0:.0f}s", flush=True)
        candidats = []
        for q in echantillon:
            hits = idx.search(q["query"], k=args.fenetre, fusion=True)
            candidats.append((q["query"], [h["id"] for h in hits], set(q["relevant"])))

    proc = demarrer_serveur(args.threads)
    lignes: list[Ligne] = []
    try:
        for n, (query, ids, pertinents) in enumerate(candidats, 1):
            t0 = time.perf_counter()
            ordre = reranger(query, [textes[i][:2000] for i in ids])
            dt = time.perf_counter() - t0
            ids_r = [ids[j] for j in ordre]
            avant = rang_premier_pertinent(ids, pertinents)
            apres = rang_premier_pertinent(ids_r, pertinents)
            lignes.append(
                Ligne(
                    query=query,
                    mots=len(query.split()),
                    n_pertinents=len(pertinents),
                    rang_avant=avant,
                    rang_apres=apres,
                    rr_avant=1 / avant if avant else 0.0,
                    rr_apres=1 / apres if apres else 0.0,
                    top10_avant=len(set(ids[:10]) & pertinents),
                    top10_apres=len(set(ids_r[:10]) & pertinents),
                    latence_s=round(dt, 1),
                )
            )
            print(f"  {n}/{len(candidats)} ({dt:.0f}s)", flush=True)
    finally:
        proc.kill()

    Path(__file__).with_name("resultats_rerank_detail.json").write_text(
        json.dumps([x._asdict() for x in lignes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Les quatre questions ------------------------------------------------
    n = len(lignes)
    gagnantes = [x for x in lignes if x.gain > 0]
    perdantes = [x for x in lignes if x.gain < 0]
    nulles = [x for x in lignes if x.gain == 0]
    print(f"\n{'=' * 62}\nDÉTAIL SUR {n} REQUÊTES\n{'=' * 62}")
    print(
        f"MRR : {statistics.mean(x.rr_avant for x in lignes):.4f} -> "
        f"{statistics.mean(x.rr_apres for x in lignes):.4f}   |   "
        f"top10 : {sum(x.top10_avant for x in lignes)} -> "
        f"{sum(x.top10_apres for x in lignes)} documents pertinents"
    )

    print(
        f"\nA. DISTRIBUTION — {len(gagnantes)} améliorées, {len(perdantes)} dégradées, "
        f"{len(nulles)} inchangées"
    )
    tri = sorted((x.gain for x in lignes), reverse=True)
    part_top20 = sum(tri[: max(1, n // 5)]) / max(1e-9, sum(g for g in tri if g > 0))
    print(
        f"   les 20 % de requêtes les plus améliorées portent "
        f"{100 * part_top20:.0f} % du gain brut total"
    )

    print(f"\nB. RÉGRESSIONS — {len(perdantes)}/{n} ({100 * len(perdantes) / n:.0f} %)")
    for x in sorted(perdantes, key=lambda z: z.gain)[:5]:
        print(f"   rang {x.rang_avant} -> {x.rang_apres} : {x.query[:64]}")

    print("\nC. PROFIL")
    for nom, groupe in (("gagnantes", gagnantes), ("perdantes", perdantes)):
        if groupe:
            print(
                f"   {nom:10s} : {statistics.mean(x.mots for x in groupe):.1f} mots, "
                f"rang avant médian "
                f"{statistics.median(x.rang_avant or 999 for x in groupe):.0f}, "
                f"{statistics.mean(x.n_pertinents for x in groupe):.1f} pertinents"
            )

    print("\nD. FENÊTRE UTILE — d'où viennent les documents promus")
    promus = [x.rang_avant for x in gagnantes if x.rang_avant is not None]
    if promus:
        for seuil in (10, 20, 30, 50):
            part = sum(1 for r in promus if r <= seuil) / len(promus)
            print(f"   rang d'origine <= {seuil:2d} : {100 * part:3.0f} % des gains")
    sans_reponse = sum(1 for x in lignes if x.rang_avant is None)
    print(
        f"   requêtes sans aucun pertinent dans la fenêtre {args.fenetre} : "
        f"{sans_reponse}/{n} (plafond structurel du rerank)"
    )
    print(
        f"\nlatence : médiane {statistics.median(x.latence_s for x in lignes):.0f}s, "
        f"min {min(x.latence_s for x in lignes):.0f}s, "
        f"max {max(x.latence_s for x in lignes):.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
