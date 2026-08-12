"""Atlas sémantique — assignation APPRISE token→cellule, canal de rappel de la fusion.

Le hachage SHA place chaque token à un emplacement aléatoire de la grille : la carte
rendue est une interface, pas un mécanisme (prouvé au bit près par le test de
permutation, chantier #366). L'atlas apprend l'assignation : une carte auto-organisée
(SOM/Kohonen, déterministe) sur les profils de cooccurrence place les tokens voisins
par le sens sur des cellules voisines — un document devient une carte de chaleur sur
un atlas de concepts.

Ce module porte la machinerie VALIDÉE du chantier de recherche #367
(research/atlas_som.py, étapes 1→3) :
- le canal atlas fusionné au trio RRF gagne +2,84 pts de R@10 et +3,65 de MRR sur le
  banc Alloprof COMPLET (2 556 docs, 2 316 requêtes réelles) — ses erreurs sont
  décorrélées de celles de la grille plate, la fusion l'exploite ;
- la convolution (flou gaussien) n'aide JAMAIS en terrain hostile : les cartes sont
  comparées brutes (σ=0) ;
- la pyramide (préfiltre par grossissement) est une piste MORTE mesurée (le contrôle
  permuté suit la même courbe : le peu qui marche vient de la projection, pas de la
  localité) — aucun préfiltre ici.

Tout est déterministe à graine fixe, CPU, sans LLM. Le coût est au BUILD : la SVD
randomisée et la SOM travaillent sur les profils du vocabulaire entier (~4 Go de pic
RAM et ~20 min à 72k tokens — chemins par tranches au-dessus de 30k). C'est pourquoi
l'atlas est opt-in (`--atlas`), jamais un défaut.
"""

import numpy as np

COTE = 64  # atlas 64×64 — la seule taille validée par la mesure (32×32 : collisions)
DIM_SOM = 128  # réduction SVD des profils avant SOM (la qualité des features est
# DÉCISIVE : SVD 0.92 vs projection JL brute 0.75 R@1, mesuré étape 1)
ITERATIONS = 30
SIGMA_DEBUT, SIGMA_FIN = 8.0, 0.8  # voisinage gaussien décroissant (en cellules)
GRAINE = 42
_TRANCHE = 8192  # au-delà de ~30k tokens, les intermédiaires pleins coûtent des Go
_SEUIL_TRANCHES = 30_000


def _normaliser(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def projeter(mat: np.ndarray, k: int = DIM_SOM, graine: int = GRAINE) -> np.ndarray:
    """SVD RANDOMISÉE seedée (Halko et al. 2011) : les composantes principales au coût
    mémoire d'une projection. Au-dessus de _SEUIL_TRANCHES lignes, les deux produits
    s'accumulent par tranches — même mathématique, pic RAM ÷2 (la copie float32
    intégrale des profils coûterait ~3,5 Go à 72k tokens). Déterministe à chemin
    constant : les sommes par tranches peuvent différer du chemin plein sur des
    décimales inertes."""
    rng = np.random.default_rng(graine)
    omega = rng.standard_normal((mat.shape[1], min(2 * k, mat.shape[1]))).astype(
        np.float32
    )
    if len(mat) <= _SEUIL_TRANCHES:
        m32 = mat.astype(np.float32)
        q, _r = np.linalg.qr(m32 @ omega)
        b = q.T @ m32
    else:
        y = np.empty((len(mat), omega.shape[1]), dtype=np.float32)
        for i in range(0, len(mat), _TRANCHE):
            y[i : i + _TRANCHE] = mat[i : i + _TRANCHE].astype(np.float32) @ omega
        q, _r = np.linalg.qr(y)
        b = np.zeros((q.shape[1], mat.shape[1]), dtype=np.float32)
        for i in range(0, len(mat), _TRANCHE):
            b += q[i : i + _TRANCHE].T @ mat[i : i + _TRANCHE].astype(np.float32)
    ub, s, _vt = np.linalg.svd(b, full_matrices=False)
    return ((q @ ub[:, :k]) * s[:k]).astype(np.float32)


def som(features: np.ndarray, graine: int = GRAINE) -> np.ndarray:
    """SOM par lots, déterministe — rend la cellule (index plat 0..COTE²-1) de chaque
    ligne de `features`. Accumulation par tranches au-dessus de _SEUIL_TRANCHES lignes
    (les intermédiaires (T, C) float64 pèseraient ~2,4 Go ×2 à 72k tokens)."""
    cellules = COTE * COTE
    rng = np.random.default_rng(graine)
    feats = _normaliser(features)
    protos = feats[rng.choice(len(feats), size=cellules, replace=len(feats) < cellules)]
    yy, xx = np.meshgrid(np.arange(COTE), np.arange(COTE), indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # (C, C)
    for it in range(ITERATIONS):
        sigma = SIGMA_DEBUT * (SIGMA_FIN / SIGMA_DEBUT) ** (it / max(1, ITERATIONS - 1))
        nprot_t = _normaliser(protos).T
        if len(feats) <= _SEUIL_TRANCHES:
            bmu = np.argmax(feats @ nprot_t, axis=1)
            vois = np.exp(-d2[bmu] / (2.0 * sigma * sigma))  # (T, C)
            masse = vois.sum(axis=0)
            masse[masse == 0] = 1.0
            protos = (vois.T @ feats) / masse[:, None]
        else:
            masse = np.zeros(cellules, dtype=np.float64)
            protos_acc = np.zeros_like(protos, dtype=np.float64)
            for i in range(0, len(feats), _TRANCHE):
                fchunk = feats[i : i + _TRANCHE]
                bmu_c = np.argmax(fchunk @ nprot_t, axis=1)
                vois = np.exp(-d2[bmu_c] / (2.0 * sigma * sigma))
                masse += vois.sum(axis=0)
                protos_acc += vois.T @ fchunk
            masse[masse == 0] = 1.0
            protos = (protos_acc / masse[:, None]).astype(feats.dtype)
    nprot_t = _normaliser(protos).T
    if len(feats) <= _SEUIL_TRANCHES:
        return np.argmax(feats @ nprot_t, axis=1)
    return np.concatenate(
        [
            np.argmax(feats[i : i + _TRANCHE] @ nprot_t, axis=1)
            for i in range(0, len(feats), _TRANCHE)
        ]
    )


def construire_mapping(profiles) -> np.ndarray:
    """Cellule de chaque token du vocabulaire, ALIGNÉE sur l'ordre de `profiles.rows`
    (le même ordre que vocab.msev) — int32, prêt pour atlas.msat."""
    v = len(profiles.rows)
    features = projeter(profiles.acc[:v])
    return som(features).astype(np.int32)


def carte(tokens: list[str], rows: dict, positions: np.ndarray, idf) -> np.ndarray:
    """Carte de chaleur tf×idf d'une liste de tokens sur l'atlas — vecteur plat
    (COTE²,) float32. Les tokens hors vocabulaire du build ne contribuent pas (la SOM
    n'est pas incrémentale — dérive observable via stats(), re-placée au rebuild)."""
    m = np.zeros(COTE * COTE, dtype=np.float32)
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    for t, c in tf.items():
        i = rows.get(t)
        if i is not None and i < len(positions):
            m[positions[i]] += c * idf(t)
    return m
