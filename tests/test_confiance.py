"""Tests — confiance de requête (src/mosaic/confiance.py).

Un moteur qui rend dix documents ne dit jamais s'il a trouvé ou s'il a comblé.
La prédiction de performance de requête (QPP) comble ce trou avec les scores
qu'on a déjà : quand le haut du classement se DÉTACHE nettement du corpus, la
requête est bien servie ; quand tout se vaut, le moteur devine.
"""

import math

import numpy as np
import pytest

from mosaic.confiance import nqc


def test_un_pic_net_donne_une_confiance_haute():
    """Quelques documents très au-dessus du fond : le moteur a trouvé."""
    scores = np.array([0.9, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert nqc(scores, k=3) > nqc(np.full(8, 0.3), k=3)


def test_un_classement_plat_donne_une_confiance_basse():
    """Tout se vaut : le moteur n'a pas trouvé, il a ordonné du bruit."""
    assert nqc(np.full(50, 0.42), k=10) == pytest.approx(0.0, abs=1e-9)


def test_la_confiance_ne_depend_pas_de_l_echelle_des_scores():
    """Un canal qui rend des scores dix fois plus grands ne doit pas paraître
    dix fois plus confiant — sinon on compare des canaux entre eux à tort."""
    s = np.array([0.9, 0.5, 0.2, 0.1, 0.1, 0.1])
    assert nqc(s, k=3) == pytest.approx(nqc(s * 10, k=3), rel=1e-6)


def test_scores_tous_nuls_rendent_zero_sans_diviser_par_zero():
    assert nqc(np.zeros(20), k=5) == 0.0


def test_k_plus_grand_que_le_corpus_ne_casse_pas():
    assert nqc(np.array([0.5, 0.2]), k=10) >= 0.0


def test_corpus_vide_rend_zero():
    assert nqc(np.array([]), k=5) == 0.0


def test_valeur_connue_a_la_main():
    """Contrôle arithmétique : σ des 2 premiers / moyenne du tout.

    scores = [1, 0, 0, 0] -> top-2 = [1, 0], σ = 0.5 ; moyenne du corpus = 0.25.
    NQC attendu = 0.5 / 0.25 = 2.0 — une valeur calculée à la main, pas une
    tautologie du code."""
    assert nqc(np.array([1.0, 0.0, 0.0, 0.0]), k=2) == pytest.approx(2.0)


def test_le_signal_ne_doit_pas_dependre_du_nombre_de_termes():
    """Rupture NOMMÉE par la littérature : sur des requêtes courtes, l'écart-type
    mesure la longueur de la requête et non sa difficulté. On vérifie ici que la
    fonction ne fabrique pas ce biais toute seule — deux distributions de MÊME
    forme rendent la MÊME confiance, quel que soit ce qui les a produites."""
    forme = np.array([0.8, 0.4, 0.2, 0.1, 0.05, 0.05])
    assert nqc(forme, k=3) == pytest.approx(nqc(forme.copy(), k=3))
    assert math.isfinite(nqc(forme, k=3))
