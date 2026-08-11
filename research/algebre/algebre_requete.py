"""Prototype — ALGÈBRE DE REQUÊTE par connecteurs (idée d'Alex).

Une requête n'est pas un sac de mots : les connecteurs (« et/ou/avec » = renforcer,
« sans/pas/ni » = soustraire, « mais pas » = contraste négatif) sont des OPÉRATEURS sur les
vecteurs des mots pleins. On compose donc deux vecteurs — q+ (ce qu'on veut) et q− (ce qu'on
exclut) — et un candidat est scoré : score = cos(cand, q+) − λ·cos(cand, q−).

But du proto : montrer, chiffres à l'appui, que le score d'un concept EXCLU descend (voire passe
négatif) pendant que les concepts voulus montent — de façon DÉTERMINISTE et EXPLICABLE, sans LLM.
Le seul point dur est la PORTÉE (quoi est nié) ; sur une requête courte et explicite, des règles
suffisent. Substrat : vecteurs-mots de la table potion de Mosaic.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)
from mosaic.embeddings import Embeddings

TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data_externes",
    "potion_fr_abtt2.msee",
)

NEG = {"sans", "pas", "ni", "sauf", "hormis", "excepté", "excepte"}
POS_RESET = {"et", "ou", "avec", "plus"}


def analyser(requete: str) -> list[tuple[str, int]]:
    """Attribue un signe (+1 vouloir / -1 exclure) à chaque mot plein, selon les connecteurs.
    Règle de portée (courte requête) : un marqueur de négation bascule en négatif ; « mais pas »
    bascule en négatif ; « mais » seul re-positive ; « et/ou/avec » gardent le signe courant."""
    toks = requete.lower().replace(",", " ").split()
    signe = 1
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "mais":
            if i + 1 < len(toks) and toks[i + 1] in NEG:
                signe = -1
                i += 2
                continue
            signe = 1
        elif t in NEG:
            signe = -1
        elif t in POS_RESET:
            pass  # garde le signe courant
        else:
            out.append((t, signe))
        i += 1
    return out


def _vec(emb, mot):
    v = emb.raw_vector(mot)
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def composer(emb, requete: str):
    """Renvoie (q_plus, q_moins, trace) — vecteurs normalisés + détail des signes/mots connus."""
    trace = []
    qp = np.zeros(emb.dim, dtype=np.float32)
    qm = np.zeros(emb.dim, dtype=np.float32)
    for mot, signe in analyser(requete):
        v = _vec(emb, mot)
        connu = v is not None
        trace.append((mot, signe, connu))
        if connu:
            (qp if signe > 0 else qm)[:] += v

    def _norm(x):
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x

    return _norm(qp), _norm(qm), trace


def score(emb, candidat: str, qp, qm, lam=0.7):
    v = _vec(emb, candidat)
    if v is None:
        return None
    s_plus = float(v @ qp)
    s_moins = float(v @ qm) if np.any(qm) else 0.0
    return round(s_plus - lam * s_moins, 4)


def demo():
    emb = Embeddings.load(Path(TABLE), abtt=2)
    candidats = [
        "disjoncteur",
        "protection",
        "tableau",
        "différentiel",
        "parafoudre",
        "domotique",
        "variateur",
        "armoire",
        "câble",
        "sécurité",
    ]
    requetes = [
        "disjoncteur",
        "disjoncteur et protection",
        "disjoncteur sans protection",
        "protection mais pas domotique",
        "tableau et parafoudre sans variateur",
    ]
    print(
        "# Algèbre de requête — score par candidat (λ=0.7). q+ renforce, q− soustrait.\n"
    )
    entete = (
        "candidat".ljust(14) + " | " + " | ".join(r[:26].ljust(26) for r in requetes)
    )
    print(entete)
    print("-" * len(entete))
    tables = [composer(emb, r) for r in requetes]
    for c in candidats:
        cells = []
        for qp, qm, _ in tables:
            s = score(emb, c, qp, qm)
            cells.append(("  n/a" if s is None else f"{s:+.3f}").ljust(26))
        print(c.ljust(14) + " | " + " | ".join(cells))
    print("\n# Lecture de la portée (mot → signe) sur deux requêtes :")
    for r in ("disjoncteur sans protection", "protection mais pas domotique"):
        _, _, tr = composer(emb, r)
        print(
            f"  « {r} » → "
            + ", ".join(
                f"{m}{'?' if not k else ''}[{'+' if s > 0 else '−'}]" for m, s, k in tr
            )
        )


if __name__ == "__main__":
    demo()
