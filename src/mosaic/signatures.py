"""Signatures ternaires creuses, déterministes par hash — aucune coordination requise."""

import hashlib

import numpy as np

K_ACTIVE = 20  # nombre de +1 (et autant de −1)


def signature(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    pos = rng.choice(dim, size=2 * K_ACTIVE, replace=False)
    sig = np.zeros(dim, dtype=np.int32)
    sig[pos[:K_ACTIVE]] = 1
    sig[pos[K_ACTIVE:]] = -1
    return sig
