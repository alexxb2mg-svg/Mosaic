"""Le canal grammatical apporte-t-il quelque chose que la grille ne trouve pas ?

C'est la brique la plus ORIGINALE du moteur — les classes fermées du français, les
négateurs, les prépositions de position, la voix passive — et elle n'a jamais été
confrontée à un banc. Elle est volontairement exclusive avec fusion/rerank/type/récence
(« le canal structural se mesure seul avant de se composer ») : ce moment n'était jamais
venu.

LA BONNE QUESTION N'EST PAS SA PERFORMANCE, C'EST SA COMPLÉMENTARITÉ. ArguAna a montré
le 13/08 qu'un canal ne vaut pas par son score moyen mais par le fait qu'il échoue à
d'AUTRES moments que les autres : la fusion y gagne +32,9 pts parce que BM25 tient là
où la grille s'écroule. Un canal grammatical qui trouverait exactement les mêmes
documents que la grille, même très bien, ne servirait à rien.

On mesure donc trois choses : son rappel seul, celui de la grille seule, et surtout le
RECOUVREMENT — combien de requêtes ratées par la grille sont rattrapées par lui.

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (13/08, 22h50)

  P-G1 — seul, il est nettement plus faible que la grille : moins de la moitié de son
         rappel. Il encode la STRUCTURE de l'énoncé, pas son sujet, et ces requêtes
         portent sur des sujets (« process CONSUEL 2026 »).

  P-G2 — mais il rattrape au moins une requête que la grille rate. Si le recouvrement
         est TOTAL — zéro rattrapage — alors il est redondant et son coût de stockage
         n'est pas justifié, quelle que soit son élégance.

  P-G3 — le banc risque d'être AVEUGLE à sa vraie valeur : il vise les négations et les
         relations de position, et ces 40 requêtes n'en contiennent probablement
         aucune. On les compte AVANT de conclure — un canal jugé sur un terrain qui
         n'exerce pas sa fonction n'est pas jugé, il est calomnié.

Usage : python research/canal_grammatical.py
        python research/canal_grammatical.py --corpus <dossier> --verite <verite.jsonl>

MESURE 2 — ALLOPROF (14/08). La première mesure (corpus interne, 40 requêtes,
recouvrement TOTAL) ne condamne pas la brique : 40 requêtes dont 6 seulement
exerçaient la fonction. Alloprof (2 316 requêtes de vrai français d'élèves) est le
terrain que ce canal vise. PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 00h50) :

  P-G1-allo — le grammatical seul reste sous 50 % du rappel de la grille (il encode
              la structure de l'énoncé, pas son sujet — vrai à toute échelle).
  P-G2-allo — à 2 316 requêtes, il rattrape AU MOINS une requête que la grille rate.
  P-G3-allo — la matière grammaticale (négation, position, passif) est présente dans
              plus de 10 % des requêtes d'élèves (le français scolaire en regorge).

CRITÈRE DE DÉCISION, déclaré avant : le canal ne justifie une place en FUSION que si
les rattrapages atteignent 1 % des requêtes (≥ 24 sur Alloprof). En dessous, il reste
hors fusion et la piste D3 (portage documenté, invitation à contribuer) est sa seule
suite — quelle que soit l'élégance de la mécanique.

VERDICT ALLOPROF (14/08, 01h20 — resultats_canal_grammatical_alloprof.json) :
banc PERTINENT (1 349/2 316 requêtes portent la matière — P-G3 fausse, dans le bon
sens) ; le canal seul fait quasi jeu égal avec la grille (0,2893 vs 0,2950 rappel —
P-G1 fausse, il est bien plus fort que prédit) ; mais 7 rattrapées contre 19 PERDUES
(P-G2 tenue de justesse, complémentarité NÉGATIVE au net). Critère 24 : ÉCHOUÉ.
=> Le canal grammatical N'ENTRE PAS en fusion. Ce n'est pas un canal de RAPPEL :
il voit le même monde que la grille, en légèrement plus flou. Sa suite est D3.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.index import Index  # noqa: E402

CORPUS = RACINE / "bench" / "_corpus_reel"
REQUETES = RACINE / "bench" / "queries.jsonl"
POTION = RACINE / "data_externes" / "potion_fr_abtt2.msee"

# Marqueurs de la matière que le canal grammatical prétend lire. Volontairement larges :
# on cherche à savoir si le banc EXERCE la fonction, pas à en mesurer la finesse.
_MATIERE = re.compile(
    r"\b(ne|n'|pas|plus|jamais|sans|aucun\w*|ni)\b"
    r"|\b(sur|sous|dans|entre|derrière|devant|au-dessus|au-dessous|avant|après)\b"
    r"|\b(est|sont|était|étaient|a été|ont été)\s+\w+(é|és|ée|ées)\b",
    re.IGNORECASE,
)


def evaluer(idx: Index, requetes: list[dict], k: int, *, grammatical: bool):
    """Rend (rappel, mrr, ensemble des requêtes RÉUSSIES) — l'ensemble sert au
    recouvrement, qui est la vraie mesure recherchée ici."""
    reussies: set[str] = set()
    rappel = mrr = 0.0
    t0 = time.perf_counter()
    for q in requetes:
        ids = [h["id"] for h in idx.search(q["query"], k=k, grammatical=grammatical)]
        pertinents = set(q["relevant"])
        touches = [i for i, d in enumerate(ids) if d in pertinents]
        part = len(set(ids) & pertinents) / len(pertinents)
        rappel += part
        if touches:
            mrr += 1.0 / (touches[0] + 1)
            reussies.add(q["query"])
    n = max(1, len(requetes))
    return {
        "rappel": round(rappel / n, 4),
        "mrr": round(mrr / n, 4),
        "duree_s": round(time.perf_counter() - t0, 1),
    }, reussies


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--verite", type=Path, default=REQUETES)
    args = ap.parse_args()

    k = 10
    requetes = [
        json.loads(li)
        for li in args.verite.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]

    # P-G3 d'abord : le banc exerce-t-il seulement la fonction qu'on juge ?
    avec_matiere = [q for q in requetes if _MATIERE.search(q["query"])]
    print(
        f"{len(requetes)} requêtes, dont {len(avec_matiere)} contiennent de la matière "
        f"grammaticale (négation, position, voix passive)"
    )
    for q in avec_matiere[:5]:
        print(f"   · {q['query'][:70]}")
    print()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        # UN SEUL index : le canal grammatical est stocké à côté de la grille, il ne la
        # modifie pas. Les deux bras interrogent donc rigoureusement le même objet.
        idx = Index.build(
            args.corpus,
            Path(tmp) / "idx",
            embeddings_path=POTION,
            abtt=2,
            weights=(0.25, 0.15, 0.60),
            rerank_vectors=True,
            type_doc=True,
            grammatical=True,
        )
        print(
            f"index construit en {time.perf_counter() - t0:.0f}s ({len(idx.ids)} docs)\n"
        )

        grille, ok_grille = evaluer(idx, requetes, k, grammatical=False)
        gram, ok_gram = evaluer(idx, requetes, k, grammatical=True)

    print(f"{'bras':20s} {'rappel':>8s} {'mrr':>8s}")
    print(f"{'grille seule':20s} {grille['rappel']:8.4f} {grille['mrr']:8.4f}")
    print(f"{'grammatical seul':20s} {gram['rappel']:8.4f} {gram['mrr']:8.4f}")

    rattrapees = ok_gram - ok_grille
    perdues = ok_grille - ok_gram
    print(
        f"\nrecouvrement : {len(ok_grille & ok_gram)} requêtes réussies par les deux · "
        f"{len(rattrapees)} RATTRAPÉES par le grammatical seul · "
        f"{len(perdues)} que lui seul rate"
    )
    for q in sorted(rattrapees)[:5]:
        print(f"   rattrapée : {q[:70]}")

    ratio = gram["rappel"] / max(1e-9, grille["rappel"])
    print(
        f"\nP-G1  grammatical / grille = {100 * ratio:.1f} % -> "
        + ("TENUE" if ratio < 0.5 else "FAUSSE")
    )
    print(
        f"P-G2  rattrapages = {len(rattrapees)} -> "
        + ("TENUE (complémentaire)" if rattrapees else "FAUSSE (redondant)")
    )
    print(
        f"P-G3  requêtes exerçant la fonction = {len(avec_matiere)}/{len(requetes)} -> "
        + (
            "TENUE (banc aveugle)"
            if len(avec_matiere) <= 2
            else "FAUSSE (banc pertinent)"
        )
    )

    suffixe = "" if args.corpus == CORPUS else f"_{args.corpus.parent.name}"
    Path(__file__).with_name(f"resultats_canal_grammatical{suffixe}.json").write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "grille": grille,
                "grammatical": gram,
                "rattrapees": len(rattrapees),
                "perdues": len(perdues),
                "requetes_avec_matiere": len(avec_matiere),
                "requetes_total": len(requetes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
