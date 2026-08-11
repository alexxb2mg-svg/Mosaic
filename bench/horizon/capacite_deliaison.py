"""Le test DIFFICILE : récupération par déliaison (horizon v2.0).

Question : un document a superposé K relations `lier(rôle_i, entité_i)`. On demande
« à quelles entités ce document est-il relié via le rôle R ? ». On DÉLIE
(permutation inverse) puis on compare aux signatures de toutes les entités. Les
vraies entités reliées via R doivent ressortir en tête. C'est ici que le brouillage
mord vraiment, et c'est ici que la dimension doit jouer.

Métrique = précision@vrai : parmi les entités recouvrées au top, combien sont justes.
1.0 = récupération parfaite. C'est la vraie courbe de capacité de l'option 3.
"""

import numpy as np

RNG = np.random.default_rng(0x2026080B)
DIMS = [3072, 12288, 24576, 49152, 98304]
KS = [5, 10, 20, 40, 80]
N_DOCS = 200
N_ENTITES = 300
N_ROLES = 6


def sig_dense(n, d):
    return RNG.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, d))


def precision_deliaison(d, k):
    ent = sig_dense(N_ENTITES, d)
    ent_u = ent / np.linalg.norm(ent, axis=1, keepdims=True)
    off = [(hash((r, "role")) % (d - 1)) + 1 for r in range(N_ROLES)]

    total_prec = 0.0
    n_cas = 0
    for _ in range(N_DOCS):
        # construire un doc : k relations (rôle, entité) distinctes
        rels = {}
        v = np.zeros(d, dtype=np.float32)
        for _ in range(k):
            r = int(RNG.integers(N_ROLES))
            e = int(RNG.integers(N_ENTITES))
            rels.setdefault(r, set()).add(e)
            v += np.roll(ent[e], off[r])
        vn = np.linalg.norm(v)
        v = v / vn if vn > 0 else v
        # pour chaque rôle présent, délier et mesurer la récupération
        for r, vraies in rels.items():
            delie = np.roll(v, -off[r])  # ≈ somme des entités reliées via r + bruit
            scores = ent_u @ delie
            top = np.argsort(-scores)[: len(vraies)]
            prec = len(set(top.tolist()) & vraies) / len(vraies)
            total_prec += prec
            n_cas += 1
    return total_prec / n_cas if n_cas else float("nan")


print("=== RÉCUPÉRATION PAR DÉLIAISON — précision@vrai (1.0 = parfait, le test dur)\n")
print("  D \\ K  |" + "".join(f"{k:>9}" for k in KS))
seuils = {}
for d in DIMS:
    ligne = f"{d:>8} |"
    for k in KS:
        p = precision_deliaison(d, k)
        marque = "*" if p >= 0.90 else (":" if p >= 0.70 else " ")
        ligne += f"{p:>8.3f}{marque}"
    print(ligne)
print(
    "\n* = precision >= 0.90 (récupération fiable) ; : = 0.70-0.90 (dégradé mais exploitable)"
)
print(
    "Lire chaque ligne : jusqu'à combien de relations/doc cette dimension tient avant"
)
print("que la déliaison ne recouvre plus les bonnes entités.")
