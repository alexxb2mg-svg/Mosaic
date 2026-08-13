"""ArguAna : le banc où la SIMILARITÉ EST UN PIÈGE. Que vaut Mosaic quand chercher
« ce qui ressemble » est exactement la mauvaise stratégie ?

Pourquoi ce banc, et pas un quatrième corpus du même genre. Alloprof, SciFact et le
corpus métier posent tous la même forme de problème — requête courte, pertinence =
« ce document parle du sujet ». Ils font varier la langue et le bruit, jamais la
STRUCTURE de la question. ArguAna la change : la requête est un argument complet, et
le document à trouver est celui qui le **RÉFUTE**. Le bon document parle donc du même
sujet en disant l'inverse, et le document le plus « ressemblant » est souvent le pire.

C'est le premier terrain capable de dire si Mosaic sait faire autre chose que de la
similarité — ou si toute son architecture repose sur une hypothèse qu'on n'avait
jamais eu l'occasion de mettre en défaut.

PIÈGE STRUCTUREL, vérifié et chiffré par ce script : les arguments-requêtes et les
documents viennent du même pool. Si le texte de la requête est LUI-MÊME dans le
corpus, tout moteur de similarité le remonte en tête et gaspille un rang sur un
document que la vérité terrain ne compte jamais comme pertinent. On mesure combien
de requêtes sont dans ce cas AVANT de commenter le moindre score.

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (13/08, 16h30)

  P-A1 — Mosaic est FAIBLE en absolu : moins de 0.40 de rappel@10. La structure est
         adverse à ce que fait le moteur ; s'il tenait ici, c'est notre compréhension
         de sa mécanique qui serait à revoir.

  P-A2 — mais c'est le premier banc où Mosaic atteint plus de **90 %** du score BM25.
         Rapport mesuré ailleurs : 67 % sur Alloprof (français bruité), 86 % sur
         SciFact (anglais propre). Ici BM25 est handicapé par la même structure — la
         littérature BEIR le donne autour de 0.31 nDCG@10, un de ses plus mauvais
         terrains. Deux moteurs également désarmés, donc écart resserré.

  P-A3 — la fusion 4 canaux apporte MOINS ici qu'ailleurs : moins de 3 points, contre
         +15 sur Alloprof et +5 sur le corpus métier. Le canal ajouté qui porte le
         gain habituel est BM25, précisément le plus mal armé pour cette tâche.

Usage : python research/arguana.py [--top 10] [--limite N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "bench"))

from bm25 import BM25  # noqa: E402  (bench/, hors paquet)

from mosaic.index import Index  # noqa: E402
from mosaic.tokenize import tokenize  # noqa: E402

CORPUS = RACINE / "bench" / "arguana" / "corpus"
VERITE = RACINE / "bench" / "arguana" / "verite.jsonl"
POTION = RACINE / "data_externes" / "potion_fr_abtt2.msee"


def batir(sortie: Path, *, canaux: bool) -> Index:
    """Le profil de production, aux canaux près. `index_paths=False` : les noms de
    fichiers BEIR sont des identifiants opaques, les indexer n'apporterait que du bruit
    (même geste que sur SciFact)."""
    return Index.build(
        CORPUS,
        sortie,
        embeddings_path=POTION,
        abtt=2,
        weights=(0.25, 0.15, 0.60),
        rerank_vectors=True,
        index_paths=False,
        hybride=canaux,
        atlas=canaux,
    )


def evaluer(chercher, requetes: list[dict], k: int, jumeau: dict[str, str]) -> dict:
    """`jumeau` : requête -> document du corpus qui EST cette requête. On demande k+1
    résultats et on retire ce jumeau, sinon les 1 401 requêtes gaspillent toutes leur
    première place sur un document que la vérité terrain ne compte jamais. C'est la
    pratique standard sur ArguAna ; sans elle, nos scores ne seraient comparables ni à
    la littérature ni à eux-mêmes d'un jeu à l'autre."""
    rappel = mrr = 0.0
    t0 = time.perf_counter()
    for q in requetes:
        exclu = jumeau.get(q["query"].strip())
        ids = [d for d in chercher(q["query"], k + 1) if d != exclu][:k]
        pertinents = set(q["relevant"])
        touches = [i for i, d in enumerate(ids) if d in pertinents]
        rappel += len(set(ids) & pertinents) / len(pertinents)
        if touches:
            mrr += 1.0 / (touches[0] + 1)
    n = max(1, len(requetes))
    return {
        "rappel": round(rappel / n, 4),
        "mrr": round(mrr / n, 4),
        "duree_s": round(time.perf_counter() - t0, 1),
    }


def mesurer_le_piege(requetes: list[dict]) -> tuple[dict, dict[str, str]]:
    """Combien de requêtes ont leur texte EXACT présent comme document du corpus ?
    Chaque cas coûte mécaniquement un rang à tous les moteurs de similarité."""
    textes = {}
    for p in CORPUS.glob("*.md"):
        t = p.read_text(encoding="utf-8").strip()
        textes[t.split("\n\n", 1)[-1].strip() if t.startswith("# ") else t] = p.name
    jumeau = {
        q["query"].strip(): textes[q["query"].strip()]
        for q in requetes
        if q["query"].strip() in textes
    }
    # `len(jumeau)` compterait les TEXTES distincts, pas les requêtes : le jeu contient
    # 107 requêtes dont le texte est répété à l'identique, et la table les fusionne.
    # Compter sur `requetes` — sinon on annonce 92,4 % là où la réalité est 100 %,
    # c'est-à-dire qu'on minimise soi-même le biais qu'on est en train de mesurer.
    touchees = sum(1 for q in requetes if q["query"].strip() in jumeau)
    distinctes = len({q["query"].strip() for q in requetes})
    return {
        "requetes_presentes_dans_le_corpus": touchees,
        "part": round(touchees / max(1, len(requetes)), 4),
        "textes_distincts": distinctes,
        "requetes_en_double": len(requetes) - distinctes,
    }, jumeau


def main() -> int:
    k = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
    requetes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    if "--limite" in sys.argv:
        requetes = requetes[: int(sys.argv[sys.argv.index("--limite") + 1])]

    piege, jumeau = mesurer_le_piege(requetes)
    print(
        f"piège structurel : {piege['requetes_presentes_dans_le_corpus']} requêtes sur "
        f"{len(requetes)} ({100 * piege['part']:.1f} %) sont elles-mêmes dans le corpus\n"
    )

    docs = sorted(p.name for p in CORPUS.glob("*.md"))
    bm = BM25([tokenize((CORPUS / n).read_text(encoding="utf-8")) for n in docs])

    def chercher_bm25(texte: str, k: int) -> list[str]:
        return [docs[i] for i in bm.search(tokenize(texte), k)]

    import contextlib
    import tempfile

    # `--index DIR` réutilise deux index déjà bâtis (sous-dossiers `avant` et `apres`)
    # au lieu d'en construire de neufs. Sur ce corpus la construction coûte 11 minutes
    # et 2,4 Go par index : une évaluation interrompue ne doit pas la refaire payer.
    reutilise = "--index" in sys.argv
    dossier = Path(sys.argv[sys.argv.index("--index") + 1]) if reutilise else None

    with (
        contextlib.nullcontext(dossier)
        if reutilise
        else tempfile.TemporaryDirectory() as tmp
    ):
        base = Path(str(tmp))
        t0 = time.perf_counter()
        if reutilise:
            avant, apres = Index.open(base / "avant"), Index.open(base / "apres")
            print(
                f"2 index rouverts en {time.perf_counter() - t0:.0f}s ({len(docs)} docs)\n"
            )
        else:
            avant = batir(base / "avant", canaux=False)
            apres = batir(base / "apres", canaux=True)
            print(
                f"2 index construits en {time.perf_counter() - t0:.0f}s ({len(docs)} docs)\n"
            )

        bras = {
            "BM25": evaluer(chercher_bm25, requetes, k, jumeau),
            "Mosaic défauts": evaluer(
                lambda t, k: [h["id"] for h in avant.search(t, k=k)],
                requetes,
                k,
                jumeau,
            ),
            "Mosaic fusion 4c": evaluer(
                lambda t, k: [h["id"] for h in apres.search(t, k=k, fusion=True)],
                requetes,
                k,
                jumeau,
            ),
        }

    print(f"{'bras':20s} {'rappel':>8s} {'mrr':>8s} {'durée':>8s}")
    for nom, r in bras.items():
        print(f"{nom:20s} {r['rappel']:8.4f} {r['mrr']:8.4f} {r['duree_s']:7.1f}s")

    bm25, defauts, fusion = (
        bras["BM25"],
        bras["Mosaic défauts"],
        bras["Mosaic fusion 4c"],
    )
    rapport = 100 * defauts["rappel"] / max(1e-9, bm25["rappel"])
    gain = 100 * (fusion["rappel"] - defauts["rappel"])
    print(
        f"\nP-A1  Mosaic = {defauts['rappel']:.4f} -> "
        + ("TENUE" if defauts["rappel"] < 0.40 else "FAUSSE")
    )
    print(
        f"P-A2  Mosaic / BM25 = {rapport:.1f} % -> "
        + ("TENUE" if rapport > 90 else "FAUSSE")
        + "   (rappel : 67 % Alloprof, 86 % SciFact)"
    )
    print(
        f"P-A3  gain de la fusion = {gain:+.2f} pts -> "
        + ("TENUE" if gain < 3.0 else "FAUSSE")
        + "   (+15 Alloprof, +5 corpus métier)"
    )

    Path(__file__).with_name("resultats_arguana.json").write_text(
        json.dumps({"piege": piege, "bras": bras}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
