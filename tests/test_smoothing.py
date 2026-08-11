import numpy as np

from mosaic.smoothing import smooth


def test_smooth_rang_plein_reconstruit():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(20, 50)).astype(np.float32)
    assert np.allclose(smooth(m, 20), m, atol=1e-3)


def test_smooth_generalise_la_structure_latente():
    # matrice bloc-structurée bruitée : 2 groupes de lignes partageant chacun une base ;
    # après lissage k=2, la similarité intra-groupe augmente, l'inter-groupe n'augmente pas plus.
    rng = np.random.default_rng(4)
    base_a, base_b = rng.normal(size=(2, 200)).astype(np.float32)
    rows = np.array(
        [base_a + 0.8 * rng.normal(size=200) for _ in range(10)]
        + [base_b + 0.8 * rng.normal(size=200) for _ in range(10)],
        dtype=np.float32,
    )

    def _mean_cos(m, i, j):
        n = m / np.linalg.norm(m, axis=1, keepdims=True)
        return float((n[i] @ n[j].T).mean())

    avant_intra = _mean_cos(rows, slice(0, 10), slice(0, 10))
    lisse = smooth(rows, 2)
    apres_intra = _mean_cos(lisse, slice(0, 10), slice(0, 10))
    assert apres_intra > avant_intra + 0.1


def test_smooth_deterministe():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(30, 80)).astype(np.float32)
    a = smooth(m, 5)
    b = smooth(m, 5)
    assert np.array_equal(a, b)


def test_rank_zero_est_identite():
    rng = np.random.default_rng(6)
    m = rng.normal(size=(10, 40)).astype(np.float32)
    out = smooth(m, 0)
    assert out is m


def test_smooth_dtype_float32():
    rng = np.random.default_rng(7)
    m = rng.normal(size=(15, 60)).astype(np.float32)
    out = smooth(m, 3)
    assert out.dtype == np.float32
    assert out.shape == m.shape


def test_smooth_graine_differente_donne_resultat_different():
    rng = np.random.default_rng(8)
    m = rng.normal(size=(30, 80)).astype(np.float32)
    a = smooth(m, 5, seed=1)
    b = smooth(m, 5, seed=2)
    assert not np.array_equal(a, b)
