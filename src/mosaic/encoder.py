"""Encodage d'un document : superposition TF-IDF des vecteurs de mots, quantifiée int8.

La négation inverse uniquement la composante signature du mot nié (spec §3.5) :
« pas conforme » ajoute normaliser(−0.5·ŝ(conforme) + 0.5·p̂(conforme)) — l'identité
s'oppose, le contexte appris reste positif.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from mosaic.tokenize import STOPWORDS

if TYPE_CHECKING:
    from mosaic.embeddings import Embeddings
    from mosaic.profiles import Profiles

NEGATORS = frozenset("pas non sans aucun aucune jamais ni not no without never".split())

# Source unique des poids (a, b, g) = (signature, profil, embedding).
# Optimum mesuré au bench v1.1 (semantique 0.250 vs 0.083 à 0.40).
WEIGHTS_DEFAULT = (0.25, 0.15, 0.60)


def _signed_counts(tokens: list[str]) -> dict[tuple[str, bool], int]:
    """Occurrences par (mot, nié) : un négateur inverse le prochain mot plein.

    Affirmé et nié sont désormais des vecteurs différents (signature opposée,
    profil partagé) — ils ne doivent pas s'annuler par sommation nette.
    """
    counts: dict[tuple[str, bool], int] = {}
    negate_next = False
    for t in tokens:
        if t in NEGATORS:
            negate_next = True
            continue
        if t in STOPWORDS:
            continue
        key = (t, negate_next)
        counts[key] = counts.get(key, 0) + 1
        negate_next = False
    return counts


def doc_channel(
    tokens: list[str], profiles: Profiles, embeddings: Embeddings, grid_dim: int
) -> np.ndarray | None:
    """Canal document (niveau SUJET, pas niveau mot) : D̂(doc) unitaire dans l'espace grille.

    D(doc)  = normaliser( Σ_t idf(t) · e_brut(t) )   — espace BRUT de la table (dim source)
    D̂(doc) = normaliser( projeter(D(doc)) )          — projection JL existante vers la grille

    Tokens **uniques** (pas de tf : le thème n'est pas la répétition), hors STOPWORDS et
    NEGATORS (la négation est locale — canal mot γ la porte déjà, le thème ne l'est pas).
    Mots hors table : ignorés (les canaux mot les couvrent). Ordre d'itération trié
    (déterminisme indépendant du hash-seed des sets Python). None si aucun token du
    document n'est dans la table, ou si la somme pondérée s'annule.
    """
    unique = sorted({t for t in tokens if t not in STOPWORDS and t not in NEGATORS})
    acc: np.ndarray | None = None
    for token in unique:
        raw = embeddings.raw_vector(token)
        if raw is None:
            continue
        contrib = np.float32(profiles.idf(token)) * raw
        acc = contrib if acc is None else acc + contrib
    if acc is None:
        return None
    doc_norm = float(np.linalg.norm(acc))
    if doc_norm == 0.0:
        return None
    d_doc = acc / np.float32(doc_norm)
    projected = embeddings.project_raw(d_doc, grid_dim)
    proj_norm = float(np.linalg.norm(projected))
    if proj_norm == 0.0:
        return None
    return projected / np.float32(proj_norm)


def quantize(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantifie un vecteur flottant en int8 par mise à l'échelle sur le pic (dernière étape
    de `encode()`, extraite pour être réutilisée telle quelle par `Index.search_like` quand un
    mélange de documents doit être re-quantifié « comme une requête » — spec v1.6 §A).

    Retourne le vecteur int8 et sa norme (0.0, vecteur nul si `v` est entièrement nul).
    """
    peak = float(np.abs(v).max())
    if peak == 0.0:
        return np.zeros(v.shape[0], dtype=np.int8), 0.0
    q = np.round(v * (127.0 / peak)).astype(np.int8)
    return q, float(np.linalg.norm(q.astype(np.float32)))


def encode(
    tokens: list[str],
    profiles: Profiles,
    embeddings: Embeddings | None = None,
    weights: tuple[float, float, float] = WEIGHTS_DEFAULT,
    doc_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    v = np.zeros(profiles.dim, dtype=np.float32)
    for (token, negated), tf in _signed_counts(tokens).items():
        weight = (1.0 + math.log(tf)) * profiles.idf(token)
        embed = (
            embeddings.vector_grid(token, profiles.dim)
            if embeddings is not None
            else None
        )
        vec = profiles.word_vector(
            token, sig_sign=-1 if negated else 1, embed=embed, weights=weights
        )
        v += np.float32(weight) * vec

    # Superposition niveau document (spec v1.3 §canal document). Le mélange (1−δ)/δ n'a de
    # sens qu'entre vecteurs UNITAIRES : on normalise v_mots d'abord, on mélange avec D̂
    # (déjà unitaire), on renormalise, PUIS on pic-quantifie — jamais l'inverse. Garde stricte :
    # doc_weight <= 0, pas de table, document vide (v_mots nul) ou canal indisponible (aucun
    # token du document dans la table) -> `v` n'est pas touché, chemin v1.2 exact reconduit
    # (aucun détour flottant, bit-identique).
    if doc_weight > 0.0 and embeddings is not None:
        v_norm = float(np.linalg.norm(v))
        if v_norm > 0.0:
            dhat = doc_channel(tokens, profiles, embeddings, profiles.dim)
            if dhat is not None:
                v_mots_unit = v / np.float32(v_norm)
                mixed = (
                    np.float32(1.0 - doc_weight) * v_mots_unit
                    + np.float32(doc_weight) * dhat
                )
                mixed_norm = float(np.linalg.norm(mixed))
                if mixed_norm > 0.0:
                    v = mixed / np.float32(mixed_norm)

    return quantize(v)
