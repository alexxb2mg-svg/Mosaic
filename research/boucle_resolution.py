"""Démo narrée de la BOUCLE DE RÉSOLUTION — le point d'orgue de Mosaic : deux moteurs qui se
complètent. Corpus SYNTHÉTIQUE (aucune donnée client), index construit puis jeté.

Le récit : deux agents ont asserté des états contradictoires pour un même chantier. La mémoire de
croyance ne devine pas — elle signale l'incertitude. Cette abstention déclenche une recherche dans
le corpus qui fait autorité ; la preuve tranche ; la croyance est ré-assérée et devient nette.
Puis un second cas où la preuve est trop mince : la boucle s'abstient, à escalader. Aucun LLM dans
la mécanique — seulement le VSA (structure) et la recherche sémantique (contexte).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.croyance import MemoireCroyance
from mosaic.index import Index
from mosaic.resolution import resoudre


def _corpus_chantier_termine(base: Path) -> Index:
    c = base / "corpus"
    c.mkdir()
    (c / "reception.md").write_text(
        "CHT_DEMO reception des travaux proces-verbal signe chantier livre client "
        "facture solde termine acheve",
        encoding="utf-8",
    )
    (c / "pv.md").write_text(
        "CHT_DEMO PV reception signe garantie parfait achevement termine livre solde final",
        encoding="utf-8",
    )
    (c / "facture.md").write_text(
        "CHT_DEMO facture finale emise chantier termine solde paiement recu cloture livre",
        encoding="utf-8",
    )
    (c / "vieux.md").write_text("CHT_DEMO pose tableau travaux cours", encoding="utf-8")
    return Index.build(
        c, base / "idx", grid=(32, 32, 3), index_paths=False, smoothing_rank=0
    )


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="mosaic_boucle_"))
    try:
        idx = _corpus_chantier_termine(base)
        print("=== BOUCLE DE RÉSOLUTION — démo (corpus synthétique) ===\n")

        print(
            "1. Deux agents assertent l'état de CHT_DEMO, en désaccord, au même instant :"
        )
        mem = MemoireCroyance(dim=512)
        mem.asserter("CHT_DEMO", "etat", "en_cours", t=5)
        mem.asserter("CHT_DEMO", "etat", "termine", t=5)
        c = mem.courant("CHT_DEMO", "etat")
        print(
            f"   -> courant() : valeur={c['valeur']}  a_preciser={c['a_preciser']}  "
            f"candidats={c.get('candidats')}"
        )
        print(f"      « {c.get('message', '')} »")
        print(
            "   La mémoire NE DEVINE PAS. L'incertitude déclenche la recherche de preuve.\n"
        )

        print(
            "2. Pour chaque état candidat, on cherche la preuve dans le corpus qui fait foi :"
        )
        form = {
            "termine": "reception proces-verbal signe livre facture solde acheve",
            "en_cours": "pose tableau travaux cours",
        }
        rap = resoudre(mem, "CHT_DEMO", "etat", idx, formulations=form)
        for p in rap["preuves"]:
            print(
                f"   - {p['valeur']:<9} évidence={p['score_preuve']:.3f}  (doc: {p['doc']})"
            )
        print()

        print(
            f"3. Verdict : {rap['statut'].upper()} -> « {rap['valeur']} » "
            f"(marge de preuve {rap['marge']:.3f}). Ré-assertion plus récente."
        )
        apres = mem.courant("CHT_DEMO", "etat")
        print(
            f"   -> courant() APRÈS : valeur={apres['valeur']}  conteste={apres['conteste']}  "
            f"a_preciser={apres['a_preciser']}"
        )
        print("   La croyance contestée est devenue NETTE, tranchée par le corpus.\n")

        print(
            "--- Contre-épreuve : quand la preuve est trop mince, la boucle s'abstient ---"
        )
        c2 = base / "corpus2"
        c2.mkdir()
        (c2 / "a.md").write_text("CHT_X etat alpha situation", encoding="utf-8")
        (c2 / "b.md").write_text("CHT_X etat beta situation", encoding="utf-8")
        idx2 = Index.build(
            c2, base / "idx2", grid=(32, 32, 3), index_paths=False, smoothing_rank=0
        )
        mem2 = MemoireCroyance(dim=512)
        mem2.asserter("CHT_X", "etat", "alpha", t=1)
        mem2.asserter("CHT_X", "etat", "beta", t=1)
        rap2 = resoudre(
            mem2,
            "CHT_X",
            "etat",
            idx2,
            formulations={"alpha": "alpha situation", "beta": "beta situation"},
        )
        print(
            f"   Verdict : {rap2['statut'].upper()} (marge {rap2['marge']:.3f}) — "
            "on N'INVENTE PAS, on escalade. La croyance reste contestée."
        )
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
