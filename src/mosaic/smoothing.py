"""Lissage rang faible des profils : SVD tronquée randomisée (Halko), numpy pur.

Généralise les associations latentes (LSA) quand le corpus est trop petit pour
faire émerger la synonymie par cooccurrence directe seule. Déterministe à
graine fixe : deux appels identiques produisent des matrices identiques.
"""

import numpy as np

SEED_DEFAULT = 0x4C534121
OVERSAMPLE = 10


def smooth(matrix: np.ndarray, rank: int, seed: int = SEED_DEFAULT) -> np.ndarray:
    """Approxime `matrix` (n×d) par sa troncature de rang `rank` (Halko randomisé).

    - G : matrice gaussienne (d × rank+10), graine fixe.
    - Y = M @ G, QR(Y) -> Q orthonormale.
    - B = Qᵀ @ M (petit facteur), SVD(B) tronquée à rang `rank`.
    - Reconstruction P_k = Q @ U_k @ diag(S_k) @ Vt_k, castée float32.

    `rank <= 0` : renvoie `matrix` inchangée (même objet, pas de copie).
    """
    if rank <= 0:
        return matrix
    n, d = matrix.shape
    m64 = matrix.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(d, rank + OVERSAMPLE))
    y = m64 @ g
    q, _ = np.linalg.qr(y)
    b = q.T @ m64
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    k = min(rank, u.shape[1])
    approx = q @ (u[:, :k] * s[:k]) @ vt[:k, :]
    return approx.astype(np.float32)
