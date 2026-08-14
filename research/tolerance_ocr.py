"""Combien coûte une faute d'OCR — et les n-grammes la rattrapent-ils ?

CONSTAT DU 14/08, sur un document réel : l'OCR d'un bon de livraison lit
« Comer » là où le logo imprime « Fournisseur ». Pour une recherche par mot exact, le
fournisseur devient introuvable par son nom sur toute une famille de scans — et
92 % des pièces comptables sont scannées.

Ce banc fait deux choses, dans cet ordre : il CHIFFRE le trou, puis il mesure si
la tolérance le comble. Deux index sont construits sur le même corpus, l'un sans
n-grammes de caractères, l'autre avec ; les mêmes requêtes leur sont posées,
saines puis abîmées par les mêmes confusions.

LES CONFUSIONS SONT CELLES D'UN OCR, PAS DES FAUTES DE FRAPPE — c'est ce qui rend
le banc honnête. Elles viennent de la ressemblance des GLYPHES imprimés (t/r, i/l,
0/O, c/e, m/n), pas de la proximité des touches.

PRÉDICTIONS DÉCLARÉES AVANT LA MESURE (14/08, 22h45, complétées à 23h05) :
  P-OCR1 — une confusion sur le mot le plus long coûte plus de 10 points de
           rappel. [MESURÉ : exactement 10,00 — prédiction FAUSSE de justesse,
           mais le phénomène est là : une requête sur dix tombe de 1,00 à 0,00.]
  P-OCR2 — la chute est plus forte sur un mot RARE que sur un mot au hasard.
           [MESURÉ : 10,00 contre 7,50 — TENUE.]
  P-NG1 — avec n-grammes, la chute est réduite d'au moins la MOITIÉ.
  P-NG2 — la contrepartie est une dilution du signal exact : le rappel sur
          requêtes SAINES ne doit pas perdre plus de 2 points, sinon on troque
          la précision de tous les jours contre la robustesse de quelques cas.

CRITÈRE D'ADOPTION : P-NG1 ET P-NG2 tenues. Un gain de robustesse payé par une
perte de précision générale ne serait pas un gain.

Usage : python research/tolerance_ocr.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from mosaic.index import Index  # noqa: E402

CORPUS = Path(os.environ.get("MOSAIC_BENCH_CORPUS", RACINE / "bench/_corpus_reel"))
VERITE = Path(os.environ.get("MOSAIC_BENCH_VERITE", RACINE / "bench/queries.jsonl"))
POTION = Path(
    os.environ.get("MOSAIC_POTION", RACINE / "data_externes/potion_fr_abtt2.msee")
)
GRAINE = 42
SANS, AVEC = "sans n-grammes", "avec n-grammes (3)"

# Confusions de GLYPHES telles qu'un OCR les produit sur du texte imprimé.
# Liste courte et vérifiable, jamais une invention statistique.
CONFUSIONS = [
    ("t", "r"),
    ("i", "l"),
    ("l", "1"),
    ("0", "O"),
    ("o", "c"),
    ("c", "e"),
    ("m", "n"),
    ("n", "h"),
    ("u", "v"),
    ("f", "t"),
    ("g", "q"),
    ("5", "S"),
    ("8", "B"),
    ("2", "Z"),
]


def abimer(mot: str, rng: random.Random) -> str:
    """Applique UNE confusion plausible, ou rend le mot intact si aucune ne
    s'applique. Jamais de suppression ni d'insertion : un OCR substitue."""
    positions = [(i, b) for i, c in enumerate(mot) for a, b in CONFUSIONS if c == a] + [
        (i, a) for i, c in enumerate(mot) for a, b in CONFUSIONS if c == b
    ]
    if not positions:
        return mot
    i, remplacement = rng.choice(positions)
    return mot[:i] + remplacement + mot[i + 1 :]


def requete_abimee(q: str, rng: random.Random, cible_rare: bool) -> tuple[str, str]:
    """Rend (requête abîmée, mot touché). `cible_rare` vise le mot le plus LONG —
    approximation simple et assumée du mot le plus discriminant."""
    mots = q.split()
    candidats = [m for m in mots if len(m) >= 4]
    if not candidats:
        return q, ""
    mot = max(candidats, key=len) if cible_rare else rng.choice(candidats)
    abime = abimer(mot, rng)
    if abime == mot:
        return q, ""
    return " ".join(abime if m == mot else m for m in mots), mot


def rappel(idx: Index, texte: str, pertinents: set[str]) -> float:
    hits = idx.search(texte, k=10, fusion=True)
    return len({h["id"] for h in hits} & pertinents) / max(1, len(pertinents))


def mesurer(idx: Index, requetes: list[dict], rng: random.Random) -> dict:
    mesures: dict[str, dict] = {}
    for nom, cible_rare in (("mot le plus long", True), ("mot au hasard", False)):
        sain = abime = 0.0
        n = degradees = 0
        exemples = []
        for q in requetes:
            pertinents = set(q["relevant"])
            texte = q["query"]
            abime_txt, mot = requete_abimee(texte, rng, cible_rare)
            if not mot:
                continue
            r_sain = rappel(idx, texte, pertinents)
            r_abime = rappel(idx, abime_txt, pertinents)
            sain += r_sain
            abime += r_abime
            n += 1
            if r_abime < r_sain:
                degradees += 1
                if len(exemples) < 3:
                    exemples.append(
                        {
                            "mot": mot,
                            "abimee": abime_txt[:52],
                            "rappel": f"{r_sain:.2f} -> {r_abime:.2f}",
                        }
                    )
        if n:
            mesures[nom] = {
                "requetes": n,
                "rappel_sain": round(sain / n, 4),
                "rappel_abime": round(abime / n, 4),
                "chute_pts": round((sain - abime) / n * 100, 2),
                "requetes_degradees": degradees,
                "exemples": exemples,
            }
            m = mesures[nom]
            print(
                f"  [{nom}] rappel {m['rappel_sain']:.4f} -> {m['rappel_abime']:.4f}"
                f"  (chute {m['chute_pts']:+.2f} pts, {degradees} requêtes dégradées)"
            )
            for e in exemples:
                print(f"      « {e['mot']} » : {e['rappel']}  — {e['abimee']}")
    return mesures


def main() -> int:
    requetes = [
        json.loads(li)
        for li in VERITE.read_text(encoding="utf-8").splitlines()
        if li.strip()
    ]
    bras: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for nom, n_gram in ((SANS, 0), (AVEC, 3)):
            t0 = time.perf_counter()
            idx = Index.build(
                CORPUS,
                Path(tmp) / f"idx{n_gram}",
                embeddings_path=POTION,
                abtt=2,
                weights=(0.25, 0.15, 0.60),
                rerank_vectors=True,
                type_doc=True,
                hybride=True,
                atlas=True,
                ngrammes=n_gram,
            )
            print(
                f"\n=== {nom} : construit en {time.perf_counter() - t0:.0f}s, "
                f"vocabulaire {len(idx.profiles.rows)} tokens",
                flush=True,
            )
            bras[nom] = mesurer(idx, requetes, random.Random(GRAINE))

    cible = "mot le plus long"
    c_sans = float(bras[SANS][cible]["chute_pts"])
    c_avec = float(bras[AVEC][cible]["chute_pts"])
    s_sans = float(bras[SANS][cible]["rappel_sain"])
    s_avec = float(bras[AVEC][cible]["rappel_sain"])

    print("\n" + "=" * 62)
    print(f"chute due à l'OCR : {c_sans:+.2f} pts  ->  {c_avec:+.2f} pts")
    print(
        f"rappel SANS faute : {s_sans:.4f}  ->  {s_avec:.4f}"
        f"  (dilution {100 * (s_avec - s_sans):+.2f} pts)"
    )
    ng1 = c_avec <= c_sans / 2
    ng2 = (s_sans - s_avec) * 100 < 2
    print("P-NG1  chute réduite de moitié -> " + ("TENUE" if ng1 else "FAUSSE"))
    print("P-NG2  dilution du sain < 2 pts -> " + ("TENUE" if ng2 else "FAUSSE"))
    verdict = "ADOPTÉ" if ng1 and ng2 else "ÉCARTÉ"
    print(f"CRITÈRE -> {verdict}")

    Path(__file__).with_name("resultats_tolerance_ocr.json").write_text(
        json.dumps({"bras": bras, "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
