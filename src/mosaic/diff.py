"""Diff sémantique — « qu'est-ce qui a changé de SENS dans ce corpus ? » (axe validé 12/08).

Deux builds déterministes du même corpus à deux moments sont comparables au bit près :
leur différence est un objet sémantique propre, pas du bruit de reconstruction — la
propriété qu'aucun moteur à embeddings-API ne peut offrir (chaque ré-indexation y bouge
tout). Mécanisme validé par banc planté (research/diff_semantique.py, P1-P3) :
spécificité STRICTE (corpus identique ⇒ diff vide, chemin d'égalité exacte), sensibilité
(mot substitué en tête des déclins d'usage, voisinage en tête de la dérive de contexte),
localité (les intacts lointains dérivent 4,5× moins que les voisins du changement).

Quatre lectures :
- INVENTAIRE : documents/vocabulaire/collocations apparus et disparus ;
- DÉRIVE DE MOT : cosinus entre les profils de cooccurrence d'un même token (dims
  ancrées par les signatures SHA — comparables entre builds) : bas = son CONTEXTE a bougé ;
- DÉRIVE D'USAGE : variation de fréquence documentaire — signal DISTINCT (un mot
  substitué garde ses contextes restants mais son df chute) ;
- DÉRIVE DE CONTEXTE : documents au contenu INTACT dont la grille a bougé — le monde a
  changé de sens autour d'eux. La lecture qu'aucun diff textuel ne donne.

Les deux index doivent partager la géométrie (grid) et l'état des tokens de chemin —
comparer deux espaces d'encodage différents produirait un delta sans signification :
refus net.
"""

import hashlib
from pathlib import Path

import numpy as np

DF_MIN = 3  # dérive de mot : sous ce df, un profil est du bruit, pas un contexte


def _hash_docs(corpus: Path) -> dict[str, str]:
    out = {}
    for p in sorted(corpus.rglob("*")):
        if p.is_file():
            out[p.relative_to(corpus).as_posix()] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    return out


def _derive(a: np.ndarray, b: np.ndarray) -> float:
    """1 − cos(a, b) avec chemin rapide d'ÉGALITÉ EXACTE → 0.0 strict : le déterminisme
    porte la garantie « identique ⇒ zéro » au sens contractuel, pas à un ulp près."""
    if np.array_equal(a, b):
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - float(a.astype(np.float64) @ b.astype(np.float64)) / (na * nb)


def diff_indexes(ia, ib, hashes_a=None, hashes_b=None, top: int = 20) -> dict:
    """Diff sémantique entre deux Index OUVERTS (états t1 et t2 du même corpus).

    `hashes_a`/`hashes_b` : empreintes de contenu par doc_id (dict id→sha) — quand
    fournies, les documents communs sont séparés en MODIFIÉS (contenu changé) et
    CONTEXTE (contenu intact dont la grille a pourtant bougé) ; sans elles, les deux
    lectures sont fusionnées sous `derive_documents`."""
    if ia.grid != ib.grid or ia.index_paths != ib.index_paths:
        raise ValueError(
            "diff refusé : les deux index n'ont pas le même espace d'encodage "
            f"(grid {ia.grid} vs {ib.grid}, path-tokens {ia.index_paths} vs "
            f"{ib.index_paths}) — comparer deux espaces différents produirait un "
            "delta sans signification ; reconstruire avec la même configuration"
        )
    docs_a, docs_b = set(ia.ids), set(ib.ids)
    vocab_a, vocab_b = set(ia.profiles.rows), set(ib.profiles.rows)
    colloc_a = set(ia.colloc) if ia.colloc else set()
    colloc_b = set(ib.colloc) if ib.colloc else set()

    va, vb = len(ia.profiles.rows), len(ib.profiles.rows)
    derive_mots: list[tuple[float, str]] = []
    for t, ra in ia.profiles.rows.items():
        rb = ib.profiles.rows.get(t)
        if rb is None or ra >= va or rb >= vb:
            continue
        if ia.profiles.df.get(t, 0) < DF_MIN or ib.profiles.df.get(t, 0) < DF_MIN:
            continue
        derive_mots.append((_derive(ia.profiles.acc[ra], ib.profiles.acc[rb]), t))
    derive_mots.sort(reverse=True)

    usage: list[tuple[float, str, int, int]] = []
    for t in vocab_a & vocab_b:
        da, db = ia.profiles.df.get(t, 0), ib.profiles.df.get(t, 0)
        if max(da, db) < DF_MIN or da == db:
            continue
        usage.append(((db - da) / max(da, db), t, da, db))
    usage.sort(key=lambda x: x[0])

    pos_a = {d: i for i, d in enumerate(ia.ids)}
    pos_b = {d: i for i, d in enumerate(ib.ids)}
    modifies: list[tuple[float, str]] = []
    contexte: list[tuple[float, str]] = []
    hashes = hashes_a is not None and hashes_b is not None
    for d in sorted(docs_a & docs_b):
        delta = _derive(ia.mat[pos_a[d]], ib.mat[pos_b[d]])
        if hashes and hashes_a.get(d) != hashes_b.get(d):
            modifies.append((delta, d))
        else:
            contexte.append((delta, d))
    modifies.sort(reverse=True)
    contexte.sort(reverse=True)

    out = {
        "docs_ajoutes": sorted(docs_b - docs_a),
        "docs_retires": sorted(docs_a - docs_b),
        "vocab_apparu": sorted(vocab_b - vocab_a),
        "vocab_disparu": sorted(vocab_a - vocab_b),
        "collocations_nees": sorted(map(str, colloc_b - colloc_a)),
        "collocations_mortes": sorted(map(str, colloc_a - colloc_b)),
        "derive_mots": [
            {"mot": t, "delta": round(d, 4)} for d, t in derive_mots[:top] if d > 0
        ],
        "derive_usage": {
            "declins": [
                {"mot": t, "ratio": round(r, 2), "df_avant": da, "df_apres": db}
                for r, t, da, db in usage[:top]
                if r < 0
            ],
            "croissances": [
                {"mot": t, "ratio": round(r, 2), "df_avant": da, "df_apres": db}
                for r, t, da, db in usage[::-1][:top]
                if r > 0
            ],
        },
    }
    if hashes:
        out["docs_modifies"] = [
            {"id": d, "delta": round(x, 4)} for x, d in modifies[:top]
        ]
        out["derive_contexte"] = [
            {"id": d, "delta": round(x, 4)} for x, d in contexte[:top] if x > 0
        ]
    else:
        out["derive_documents"] = [
            {"id": d, "delta": round(x, 4)} for x, d in contexte[:top] if x > 0
        ]
    return out


def diff_corpus(corpus_a: Path, corpus_b: Path, top: int = 20) -> dict:
    """Diff sémantique entre deux DOSSIERS de corpus : construit deux index éphémères
    (mêmes défauts, même espace) puis délègue à diff_indexes — avec les empreintes de
    contenu, qui séparent documents modifiés et dérive de contexte."""
    import tempfile

    from mosaic.index import Index

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ia = Index.build(corpus_a, Path(tmp) / "a")
        ib = Index.build(corpus_b, Path(tmp) / "b")
        return diff_indexes(
            ia,
            ib,
            hashes_a=_hash_docs(corpus_a),
            hashes_b=_hash_docs(corpus_b),
            top=top,
        )
