"""L'aiguilleur déroute-t-il la prose ? — la seule question qui décide de son sort.

P-C3, déclarée bien avant d'écrire l'aiguilleur : « le piège sera la DÉTECTION.
Un routage naïf enverrait tout à la lecture exacte et casserait la prose. C'est
là que la piste peut échouer, pas sur la lecture elle-même. »

Ce banc mesure exactement ça, sur trois populations qui n'ont pas le même rôle :

  TÉMOIN — Alloprof, 2 316 vraies questions d'élèves. AUCUNE n'est un comptage
  ni une demande d'ordre : ce sont des questions de sens. Elles doivent donc
  partir en SÉMANTIQUE. Chaque déroutement est une FAUSSE ALARME, et c'est le
  chiffre qui condamne ou absout l'aiguilleur.

  MÉTIER — les questions réellement posées au moteur (journal des recherches) :
  on regarde où elles vont, sans vérité étiquetée — c'est de l'observation.

  CIBLE — questions construites, une par circuit, avec la réponse attendue :
  vérifie que l'aiguilleur RECONNAÎT ce pour quoi il est fait (le rappel).

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 19h40) :
  P-AIG1 — moins de 2 % de fausses alarmes sur Alloprof. Au-delà, l'aiguilleur
           est trop agressif et ne doit pas être branché.
  P-AIG2 — au moins 90 % de reconnaissance sur les questions CIBLE.
  P-AIG3 — les fausses alarmes qui subsisteront viendront de « combien de »
           employé en prose (« combien de temps », « combien de fois »).

CRITÈRE D'ADOPTION : P-AIG1 ET P-AIG2 tenues. Sinon, l'aiguilleur reste un
outil de diagnostic, jamais un routeur automatique.

Usage : python research/aiguilleur_precision.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.aiguilleur import Circuit, aiguiller  # noqa: E402

VERITE = Path(
    os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/alloprof/verite.jsonl")
)
JOURNAL = Path(os.environ.get("MOSAIC_JOURNAL", ""))

# Une question par circuit, avec le circuit attendu — le rappel de l'aiguilleur.
CIBLES = [
    ("combien de bons de livraison en juin 2026", Circuit.COMPTAGE),
    ("combien d'articles dans la banque de prix", Circuit.COMPTAGE),
    ("nombre de photos sur le chantier", Circuit.COMPTAGE),
    ("quel est le nombre de devis émis cette année", Circuit.COMPTAGE),
    ("le dernier bon de livraison du fournisseur", Circuit.ORDRE),
    ("la note de chantier la plus récente", Circuit.ORDRE),
    ("les 5 derniers devis envoyés", Circuit.ORDRE),
    ("quelle est la facture la plus récente", Circuit.ORDRE),
    ("le devis DE26040008", Circuit.REFERENCE),
    ("bon de livraison 9990001", Circuit.REFERENCE),
    ("article 9990004 du catalogue", Circuit.REFERENCE),
    ("quelle marque de sèche-serviettes a été posée", Circuit.SEMANTIQUE),
    ("photo du tableau électrique avant travaux", Circuit.SEMANTIQUE),
    ("pourquoi le disjoncteur saute quand j'allume le four", Circuit.SEMANTIQUE),
]


def main() -> int:
    print("=" * 66)
    print("TÉMOIN — Alloprof : des questions de SENS, aucune n'est un comptage")
    lignes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    routes = [(q["query"], aiguiller(q["query"])) for q in lignes]
    compte = Counter(r.circuit for _, r in routes)
    deroutees = [(q, r) for q, r in routes if r.circuit is not Circuit.SEMANTIQUE]
    taux = 100 * len(deroutees) / max(1, len(routes))
    print(
        f"  {len(routes)} questions -> "
        + ", ".join(f"{c.value} {n}" for c, n in compte.most_common())
    )
    print(f"  FAUSSES ALARMES : {len(deroutees)}/{len(routes)} ({taux:.2f} %)")
    for q, r in deroutees[:8]:
        print(f"    [{r.circuit.value}] {q[:58]}  <- {r.motif}")

    print("\nCIBLE — l'aiguilleur reconnaît-il ce pour quoi il est fait ?")
    bons = 0
    for q, attendu in CIBLES:
        obtenu = aiguiller(q).circuit
        ok = obtenu is attendu
        bons += ok
        if not ok:
            print(
                f"    RATÉ  {q[:52]:52s} attendu {attendu.value}, obtenu {obtenu.value}"
            )
    rappel = 100 * bons / len(CIBLES)
    print(f"  {bons}/{len(CIBLES)} correct ({rappel:.0f} %)")

    if JOURNAL and JOURNAL.exists():
        print("\nMÉTIER — les vraies questions posées au moteur (observation)")
        vues = []
        for li in JOURNAL.read_text(encoding="utf-8").splitlines():
            if li.strip():
                try:
                    vues.append(json.loads(li)["q"])
                except (json.JSONDecodeError, KeyError):
                    continue
        for q in dict.fromkeys(vues):
            r = aiguiller(q)
            print(f"    [{r.circuit.value:10s}] {q[:56]}")

    print("\n" + "=" * 66)
    print(
        f"P-AIG1  fausses alarmes {taux:.2f} % -> "
        + ("TENUE" if taux < 2 else "FAUSSE")
    )
    print(
        f"P-AIG2  rappel cible {rappel:.0f} % -> "
        + ("TENUE" if rappel >= 90 else "FAUSSE")
    )
    verdict = "ADOPTÉ" if taux < 2 and rappel >= 90 else "DIAGNOSTIC seulement"
    print(f"CRITÈRE -> {verdict}")

    Path(__file__).with_name("resultats_aiguilleur.json").write_text(
        json.dumps(
            {
                "temoin_questions": len(routes),
                "fausses_alarmes": len(deroutees),
                "taux_fausses_alarmes_pct": round(taux, 2),
                "exemples_deroutees": [
                    {"question": q, "circuit": r.circuit.value, "motif": r.motif}
                    for q, r in deroutees[:20]
                ],
                "rappel_cible_pct": round(rappel, 1),
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
