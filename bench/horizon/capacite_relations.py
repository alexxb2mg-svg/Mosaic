"""Courbe de capacité du canal de relations (horizon v2.0, question d'Alex).

Question mesurée : dans un vecteur de dimension D, si l'on superpose K relations liées
`lier(rôle, entité)` dans la mosaïque d'un document, peut-on encore retrouver de façon
fiable si une relation précise (rôle, entité) est présente ?

Liaison = permutation circulaire déterministe (np.roll d'un décalage = hash(rôle)).
Métrique = AUC : probabilité qu'un document CONTENANT la relation ait un cosinus plus
élevé qu'un document ne la contenant pas. 1.0 = parfait, 0.5 = hasard. Le seuil pratique
de « propre » est AUC >= 0.95.

On compare deux familles de vecteurs de base :
  - DENSE bipolaire (±1 partout) — standard du calcul hyperdimensionnel pour la liaison ;
  - CREUX ternaire (40 non-nuls) — les signatures actuelles de Mosaic.
Ce comparatif dit au passage quel type de vecteur le canal relations devrait utiliser.
"""

import numpy as np

GRAINE = 0x2026080A
RNG = np.random.default_rng(GRAINE)

DIMS = [3072, 12288, 24576, 49152, 98304]
KS = [5, 10, 20, 40, 80, 160]
N_DOCS = 400
N_ENTITES = 200
N_ROLES = 8
N_REQUETES = 200  # relations testées par configuration


def sig_dense(n, d):
    """n vecteurs bipolaires ±1 de dimension d."""
    return RNG.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, d))


def sig_creux(n, d, k_actif=40):
    """n vecteurs ternaires creux (k_actif/2 à +1, k_actif/2 à -1) — signatures Mosaic."""
    m = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        pos = RNG.choice(d, size=k_actif, replace=False)
        m[i, pos[: k_actif // 2]] = 1.0
        m[i, pos[k_actif // 2 :]] = -1.0
    return m


def decalages_roles(d):
    """Décalage de permutation déterministe par rôle (hash → offset dans [1, d))."""
    return [(hash((r, "role")) % (d - 1)) + 1 for r in range(N_ROLES)]


def auc(d, k, famille):
    """AUC de détection d'une relation parmi k superposées, en dimension d."""
    ent = sig_dense(N_ENTITES, d) if famille == "dense" else sig_creux(N_ENTITES, d)
    off = decalages_roles(d)

    def lier(role, entite):
        return np.roll(ent[entite], off[role])

    # chaque document = somme normalisée de k relations (rôle, entité) tirées au hasard
    docs = np.zeros((N_DOCS, d), dtype=np.float32)
    relations_doc = []  # ensemble des (role, entite) par doc
    for i in range(N_DOCS):
        rels = set()
        v = np.zeros(d, dtype=np.float32)
        for _ in range(k):
            r = int(RNG.integers(N_ROLES))
            e = int(RNG.integers(N_ENTITES))
            rels.add((r, e))
            v += lier(r, e)
        n = np.linalg.norm(v)
        docs[i] = v / n if n > 0 else v
        relations_doc.append(rels)

    # requêtes : des relations réellement présentes dans au moins un doc
    presentes = list({rel for rels in relations_doc for rel in rels})
    RNG.shuffle(presentes)
    presentes = presentes[:N_REQUETES]

    gagnes = total = 0
    for r, e in presentes:
        q = lier(r, e)
        q = q / np.linalg.norm(q)
        cos = docs @ q
        pos = [i for i, rels in enumerate(relations_doc) if (r, e) in rels]
        neg = [i for i, rels in enumerate(relations_doc) if (r, e) not in rels]
        if not pos or not neg:
            continue
        # AUC échantillonnée : paires (positif, négatif)
        pc = cos[pos]
        nc = cos[neg]
        # comparaison vectorisée d'un échantillon de paires
        echant_neg = nc[RNG.integers(0, len(nc), size=min(len(nc), 300))]
        for cp in pc:
            gagnes += int(np.sum(cp > echant_neg))
            total += len(echant_neg)
    return gagnes / total if total else float("nan")


print("=== CAPACITÉ DU CANAL RELATIONS — AUC de détection (>= 0.95 = propre)\n")
for famille in ("dense", "creux"):
    print(f"--- vecteurs {famille.upper()}")
    entete = "  D \\ K  |" + "".join(f"{k:>8}" for k in KS)
    print(entete)
    for d in DIMS:
        ligne = f"{d:>8} |"
        for k in KS:
            a = auc(d, k, famille)
            marque = "*" if a >= 0.95 else " "
            ligne += f"{a:>7.3f}{marque}"
        print(ligne)
    print()
print("* = AUC >= 0.95 (récupération propre). Lire : à cette dimension, ce nombre de")
print("relations par document reste distinguable ; au-delà, le brouillage gagne.")
