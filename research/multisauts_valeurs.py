"""Multi-sauts VECTEUR-NATIF — parcours de graphe dans l'espace des valeurs, sans nom.

Question (Alex) : « on n'a pas besoin des mots, juste des valeurs » — peut-on suivre A→B→C
entièrement dans la grille ? La sonde permutation-par-nom exigeait de connaître le NOM de
l'étape intermédiaire (le décalage en dépend). Réponse mesurée ici en trois designs :

  1. MULTIPLICATIF PUR — bind(A,B) = sig(A) ⊙ sig(B) (produit ±1, auto-inverse). Pur-valeurs
     MAIS symétrique : en déliant s1 on récupère AUSSI son prédécesseur (A⊙B = B⊙A) →
     confusion 50/50 dès le 2e saut (mesuré ~0.5). Le produit perd la DIRECTION de l'arête.
  2. CHAÎNAGE BRUT (sans cleanup intermédiaire) — mort à 0 % dès 2 sauts, sur les DEUX
     primitives : les termes croisés de la superposition explosent. Fondamental.
  3. MULTIPLICATIF ORIENTÉ (le bon design) — bind(A→B) = sig(A) ⊙ roll(sig(B), 1). La
     permutation est FIXE et UNIVERSELLE (elle marque le rôle « successeur », elle ne dépend
     d'aucun nom). Délier les sortants de X : roll⁻¹(X ⊙ G). Cleanup par saut = produit
     matriciel + argmax contre le codebook — 100 % vectoriel.

Résultats (dim 12288, 80 essais/case, codebook 400 valeurs) :
  prof 2 : 0.99 (E=10) / 0.94 (E=30) / 0.86 (E=80)
  prof 4 : 0.99 / 0.93 / 0.75
  prof 6 : 1.00 / 0.85 / 0.57
Meilleur que la permutation-par-nom (0.90 à prof 6, E=20) ET pur-valeurs de bout en bout.

Verdict : le multi-saut « juste avec les valeurs » est VIABLE — à condition d'un cleanup
vectoriel à chaque saut (incontournable, prouvé) et d'orienter l'arête par une permutation
fixe. Prochaine marche (si un canal graphe v3 se justifie) : source d'arêtes généraliste.
"""

import numpy as np

DIM = 12288
N_SYMBOLS = 400
ESSAIS = 80


def _codebook(rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(N_SYMBOLS, DIM)).astype(
        np.float64
    )


def _chaine_et_graphe(cb: np.ndarray, rng, longueur: int, e_total: int):
    idx = rng.choice(N_SYMBOLS, size=longueur + 1, replace=False)
    aretes = [(int(idx[i]), int(idx[i + 1])) for i in range(longueur)]
    while len(aretes) < e_total:
        a, b = rng.choice(N_SYMBOLS, size=2, replace=False)
        if (int(a), int(b)) not in aretes:
            aretes.append((int(a), int(b)))
    g = np.zeros(DIM)
    for a, b in aretes:
        g += cb[a] * np.roll(cb[b], 1)  # bind orienté : A ⊙ roll(B, 1)
    return g, idx


def _cleanup(cb: np.ndarray, paquet: np.ndarray) -> int:
    """La valeur la plus proche du paquet — produit matriciel + argmax, pur vectoriel."""
    return int(np.argmax(cb @ paquet))


def taux_succes(
    longueur: int, e_total: int, essais: int = ESSAIS, seed: int = 0x4D5341
):
    rng = np.random.default_rng(seed)
    cb = _codebook(rng)
    ok = 0
    for _ in range(essais):
        g, idx = _chaine_et_graphe(cb, rng, longueur, e_total)
        cur = int(idx[0])
        bon = True
        for h in range(longueur):
            nxt = _cleanup(cb, np.roll(cb[cur] * g, -1))  # sortants de cur
            if nxt != int(idx[h + 1]):
                bon = False
                break
            cur = nxt
        ok += bon
    return ok / essais


def main() -> None:
    print(f"=== multi-sauts pur-valeurs (multiplicatif orienté, dim {DIM}) ===")
    print(f"{'prof':>4} | {'E':>4} | {'succès':>6}")
    for longueur in (1, 2, 3, 4, 6):
        for e in (10, 30, 80):
            print(f"{longueur:>4} | {e:>4} | {taux_succes(longueur, e):>6.2f}")


if __name__ == "__main__":
    main()
