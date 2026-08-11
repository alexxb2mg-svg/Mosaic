"""Test de permutation — la grille est-elle un MÉCANISME ou une INTERFACE ?

État des lieux gravé (analyse bench Alloprof, briefing #366) : si une permutation fixe
des dimensions, appliquée identiquement à tous les vecteurs documents ET à la requête,
laisse tous les classements identiques, alors AUCUN résultat de recherche ne dépend de
la géométrie 2D de la grille — les images deviennent méconnaissables, les résultats ne
bougent pas d'un rang. C'est le critère falsifiable : une future grille ORGANISÉE
(chantier atlas sémantique, briefing #367) devra faire ÉCHOUER ce test pour mériter
d'exister — via des opérateurs réellement 2D (convolution, pyramide), pas via le cosinus.

Rigueur : les scores sont calculés en int64 EXACT (les vecteurs stockés sont int8 — le
produit scalaire est un entier, invariant par permutation au bit près ; aucun artefact
d'ordre de sommation flottante ne peut brouiller le verdict). Le rendu, lui, est comparé
composante à composante : toute la structure spatiale est déplacée (la fraction rapportée
sous-estime le brouillage — les zéros du vecteur creux retombent souvent sur des zéros).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mosaic.collocations import merge
from mosaic.encoder import encode
from mosaic.index import Index
from mosaic.lexicon import canonicalize
from mosaic.tokenize import tokenize

GRID = (32, 32, 3)
DOCS = {
    "tableau.md": "remplacement tableau electrique disjoncteur differentiel protection",
    "cuisine.md": "recette sauce tomate basilic olive cuisson douce",
    "devis.md": "devis chantier peinture couloir escalier reference",
    "jardin.md": "taille haie arbuste printemps outils secateur",
    "reseau.md": "cablage reseau baie brassage rj45 switch etage",
}
REQUETES = ["disjoncteur tableau", "sauce tomate", "cablage rj45"]


def _q_exact(idx: Index, texte: str) -> np.ndarray:
    tokens = merge(
        merge(canonicalize(tokenize(texte), idx._compiled), idx.colloc), idx.colloc
    )
    q, _ = encode(tokens, idx.profiles, weights=idx.weights)
    return q.astype(np.int64)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        for nom, texte in DOCS.items():
            (corpus / nom).write_text(texte, encoding="utf-8")
        idx = Index.build(corpus, Path(tmp) / "idx", grid=GRID)

        dim = idx.mat.shape[1]
        perm = np.random.default_rng(42).permutation(dim)
        mat = idx.mat.astype(np.int64)
        mat_perm = mat[:, perm]

        classements_identiques = True
        for req in REQUETES:
            q = _q_exact(idx, req)
            s = mat @ q  # int64 exact
            s_perm = mat_perm @ q[perm]
            assert np.array_equal(s, s_perm), (
                "les scores exacts devraient être invariants"
            )
            classements_identiques &= bool(
                np.array_equal(
                    np.argsort(-s, kind="stable"), np.argsort(-s_perm, kind="stable")
                )
            )

        # le rendu, lui, est détruit : fraction de composantes de la grille qui changent
        pixels_changes = float(np.mean(idx.mat[0] != idx.mat[0][perm]))

        print("=== Test de permutation (briefing #366) ===")
        print(f"dimensions permutées : {dim} (graine 42)")
        print(f"scores int64 identiques au bit près : oui ({len(REQUETES)} requêtes)")
        print(f"classements identiques : {'oui' if classements_identiques else 'NON'}")
        print(
            f"composantes de l'image changées par la permutation : {pixels_changes:.1%}"
        )
        print()
        if classements_identiques:
            print(
                "VERDICT : la grille actuelle est une INTERFACE (rendu/diagnostic), pas un"
            )
            print(
                "mécanisme de recherche — le moteur est le vecteur plat hyperdimensionnel."
            )
            print(
                "Une grille organisée (atlas sémantique, #367) devra faire échouer CE test"
            )
            print(
                "via des opérateurs 2D réels pour prouver qu'elle apporte un mécanisme."
            )
            return 0
        print(
            "VERDICT INATTENDU : un classement dépend de l'ordre des dimensions — investiguer."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
