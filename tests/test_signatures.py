import numpy as np

from mosaic.signatures import K_ACTIVE, signature

DIM = 12288


def test_forme_et_valeurs():
    s = signature("disjoncteur", DIM)
    assert s.shape == (DIM,) and s.dtype == np.int32
    assert int((s == 1).sum()) == K_ACTIVE
    assert int((s == -1).sum()) == K_ACTIVE
    assert int((s == 0).sum()) == DIM - 2 * K_ACTIVE


def test_deterministe():
    a = signature("tension", DIM)
    b = signature("tension", DIM)
    assert np.array_equal(a, b)


def test_mots_differents_quasi_orthogonaux():
    a = signature("tension", DIM)
    b = signature("carrelage", DIM)
    # 40 positions actives sur 12288 : le produit scalaire attendu est ~0
    assert abs(int(a @ b)) <= 8


def test_dimension_compacte():
    s = signature("tension", 3072)
    assert s.shape == (3072,)
