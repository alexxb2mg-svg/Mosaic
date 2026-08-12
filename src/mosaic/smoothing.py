"""Lissage rang faible des profils : SVD tronquée randomisée (Halko), numpy pur.

Généralise les associations latentes (LSA) quand le corpus est trop petit pour
faire émerger la synonymie par cooccurrence directe seule. Déterministe à
graine fixe : deux appels identiques produisent des matrices identiques.
"""

import numpy as np

SEED_DEFAULT = 0x4C534121
OVERSAMPLE = 10


def smooth(
    matrix: np.ndarray,
    rank: int,
    seed: int = SEED_DEFAULT,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Approxime `matrix` (n×d) par sa troncature de rang `rank` (Halko randomisé).

    - G : matrice gaussienne (d × rank+10), graine fixe.
    - Y = M @ G, QR(Y) -> Q orthonormale.
    - B = Qᵀ @ M (petit facteur), SVD(B) tronquée à rang `rank`.
    - Reconstruction P_k = Q @ U_k @ diag(S_k) @ Vt_k, castée float32.

    `rank <= 0` : renvoie `matrix` inchangée (même objet, pas de copie).
    `out` (float32, même forme) : reçoit le résultat SANS matérialiser la copie
    float32 intermédiaire — la conversion se fait par blocs de lignes, cast
    élément par élément STRICTEMENT identique au astype global (aucun BLAS dans
    une conversion : le bit près est garanti). Écrire out=matrix est licite :
    `matrix` n'est plus relue après la copie float64 d'entrée.

    Discipline RAM (plafond mesuré research/ram_build.py : le lissage dominait le
    pic du build) : chaque gros intermédiaire n×d est libéré dès qu'il devient
    inutile — les OPÉRATIONS et leur ordre sont inchangés, seule la durée de vie
    des tampons change, donc le résultat est bit-identique à la version
    historique (vérifié par checksum d'index complet).
    """
    if rank <= 0:
        return matrix
    n, d = matrix.shape
    m64 = matrix.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(d, rank + OVERSAMPLE))
    y = m64 @ g
    q, _ = np.linalg.qr(y)
    del y
    b = q.T @ m64
    del m64  # libère 8·n·d octets AVANT la reconstruction (l'entrée n'est plus lue)
    u, s, vt = np.linalg.svd(b, full_matrices=False)
    del b
    k = min(rank, u.shape[1])
    w = q @ (u[:, :k] * s[:k])  # même associativité que l'expression historique
    del q, u, s
    approx = w @ vt[:k, :]
    del w, vt
    if out is None:
        return approx.astype(np.float32)
    for i in range(0, n, 4096):
        out[i : i + 4096] = approx[i : i + 4096]
    return out
