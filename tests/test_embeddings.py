import gzip

import numpy as np

from mosaic.embeddings import Embeddings, prepare

DIM_GRID = 3072


def _fake_vec_gz(tmp_path, rows):
    # format fastText : "N dim" puis "mot v1 ... v300"
    p = tmp_path / "mini.vec.gz"
    lines = [f"{len(rows)} 300"]
    for word, seed in rows:
        rng = np.random.default_rng(seed)
        vals = " ".join(f"{v:.4f}" for v in rng.normal(size=300))
        lines.append(f"{word} {vals}")
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def test_prepare_et_load_round_trip(tmp_path):
    src = _fake_vec_gz(
        tmp_path, [("chat", 1), ("chien", 2), ("Maison", 3), ("rare", 4)]
    )
    out = tmp_path / "table.msee"
    stats = prepare(src, out, keep=3)
    emb = Embeddings.load(out)
    assert stats["kept"] == 3 and emb.dim == 300
    assert emb.vector_grid("chat", DIM_GRID) is not None
    assert emb.vector_grid("maison", DIM_GRID) is not None  # minuscules
    assert emb.vector_grid("rare", DIM_GRID) is None  # au-delà de keep
    assert len(emb.sha) == 64


def test_extra_words_au_dela_de_keep(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1), ("chien", 2), ("disjoncteur", 3)])
    out = tmp_path / "table.msee"
    prepare(src, out, keep=1, extra_words={"disjoncteur"})
    emb = Embeddings.load(out)
    assert emb.vector_grid("disjoncteur", DIM_GRID) is not None
    assert emb.vector_grid("chien", DIM_GRID) is None


def test_proches_voisinage(tmp_path):
    # chat et chaton partagent la MÊME graine -> vecteur identique -> voisins l'un de l'autre.
    src = _fake_vec_gz(
        tmp_path, [("chat", 1), ("chaton", 1), ("chien", 2), ("avion", 7)]
    )
    out = tmp_path / "table.msee"
    prepare(src, out, keep=4)
    emb = Embeddings.load(out)
    voisins = emb.proches("chat", k=3)
    assert voisins is not None
    assert voisins[0][0] == "chaton"  # vecteur identique -> rang 1
    assert voisins[0][1] > 0.99  # cosinus ~1
    assert all(mot != "chat" for mot, _ in voisins)  # soi-même exclu
    assert emb.proches("mot_absent") is None  # mot hors table -> None


def test_projection_preserve_le_cosinus(tmp_path):
    src = _fake_vec_gz(tmp_path, [("a", 10), ("b", 10), ("c", 99)])  # a et b identiques
    out = tmp_path / "t.msee"
    prepare(src, out)
    emb = Embeddings.load(out)
    va, vb, vc = (emb.vector_grid(w, DIM_GRID) for w in ("a", "b", "c"))
    assert float(va @ vb) > 0.99  # vecteurs sources identiques → projetés identiques
    assert abs(float(va @ vc)) < 0.2  # gaussiens indépendants → quasi orthogonaux


def test_vector_grid_unitaire_et_cache(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1)])
    out = tmp_path / "t.msee"
    prepare(src, out)
    emb = Embeddings.load(out)
    v1 = emb.vector_grid("chat", DIM_GRID)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4
    assert emb.vector_grid("chat", DIM_GRID) is v1  # même objet → cache


def test_lignes_malformees_ignorees(tmp_path):
    """Malformed lines (wrong column count) are skipped; valid words remain."""
    p = tmp_path / "malformed.vec.gz"
    # Create a .vec.gz with 3 expected words but include a malformed line
    lines = [
        "3 300",
        "valide1 "
        + " ".join(f"{x:.4f}" for x in np.random.default_rng(10).normal(size=300)),
    ]
    lines.append("cassé 0.1 0.2")  # Only 2 values, not 300+1 → malformed
    lines.append(
        "valide2 "
        + " ".join(f"{x:.4f}" for x in np.random.default_rng(20).normal(size=300))
    )
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    out = tmp_path / "table.msee"
    stats = prepare(p, out)
    emb = Embeddings.load(out)

    # Only 2 valid words should be kept (valide1, valide2)
    assert stats["kept"] == 2
    assert emb.vector_grid("valide1", DIM_GRID) is not None
    assert emb.vector_grid("valide2", DIM_GRID) is not None
    assert emb.vector_grid("cassé", DIM_GRID) is None  # malformed line was skipped


def test_abtt_reduit_la_composante_commune(tmp_path):
    # 30 mots = base commune forte (même direction dominante) + bruit propre à chaque mot.
    rng = np.random.default_rng(42)
    base = rng.normal(size=300) * 5.0  # composante commune forte
    words = [f"mot{i}" for i in range(30)]
    lines = ["30 300"]
    for w in words:
        noise = rng.normal(size=300) * 0.3
        vec = base + noise
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in vec))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "table.msee"
    prepare(src, out)

    def _mean_cosine(emb):
        vecs = emb.mat.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / norms
        cos = unit @ unit.T
        iu = np.triu_indices(len(words), k=1)
        return float(np.mean(cos[iu]))

    emb0 = Embeddings.load(out, abtt=0)
    emb2 = Embeddings.load(out, abtt=2)
    assert _mean_cosine(emb2) < _mean_cosine(emb0) - 0.2


def test_abtt_zero_inchange(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1), ("chien", 2), ("maison", 3)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    emb0 = Embeddings.load(out, abtt=0)
    emb_default = Embeddings.load(out)
    raw = out.read_bytes()
    # relit la matrice brute directement depuis le fichier (float16) pour comparer
    from mosaic.embeddings import _HEADER
    import json as _json
    import struct as _struct

    _magic, _vmaj, _vmin, n, dim = _HEADER.unpack_from(raw, 0)
    off = _HEADER.size
    (blob_len,) = _struct.unpack_from("<Q", raw, off)
    off += 8
    words = _json.loads(raw[off : off + blob_len].decode("utf-8"))
    file_mat = np.frombuffer(
        raw, dtype=np.float16, offset=off + blob_len, count=n * dim
    ).reshape(n, dim)
    assert emb0.abtt == 0
    assert emb0.sha == emb_default.sha
    np.testing.assert_array_equal(
        emb0.mat.astype(np.float32), file_mat.astype(np.float32)
    )
    np.testing.assert_array_equal(
        emb_default.mat.astype(np.float32), file_mat.astype(np.float32)
    )
    assert words == list(emb0.rows.keys())


def test_abtt_deterministe(tmp_path):
    src = _fake_vec_gz(tmp_path, [(f"mot{i}", i) for i in range(20)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    emb_a = Embeddings.load(out, abtt=2)
    emb_b = Embeddings.load(out, abtt=2)
    assert np.array_equal(emb_a.mat, emb_b.mat)
    assert emb_a.abtt == 2 and emb_b.abtt == 2
    assert emb_a.sha == emb_b.sha


def test_raw_vector_brut_et_absent(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1), ("chien", 2)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    emb = Embeddings.load(out)
    raw = emb.raw_vector("chat")
    assert raw is not None
    assert raw.dtype == np.float32 and raw.shape == (300,)
    # brut : PAS normalisé par mot (contrairement à vector_grid)
    assert abs(float(np.linalg.norm(raw)) - 1.0) > 1e-3
    assert emb.raw_vector("inconnu") is None


def test_project_raw_reutilise_le_cache_de_projection(tmp_path):
    src = _fake_vec_gz(tmp_path, [("a", 10), ("b", 10), ("c", 99)])  # a et b identiques
    out = tmp_path / "table.msee"
    prepare(src, out)
    emb = Embeddings.load(out)
    ra, rb, rc = (emb.raw_vector(w) for w in ("a", "b", "c"))
    pa = emb.project_raw(ra, DIM_GRID)
    pb = emb.project_raw(rb, DIM_GRID)
    pc = emb.project_raw(rc, DIM_GRID)
    assert pa.shape == (DIM_GRID,)
    # même projection que vector_grid (à la normalisation près) : cosinus préservé
    cos_ab = float((pa / np.linalg.norm(pa)) @ (pb / np.linalg.norm(pb)))
    cos_ac = float((pa / np.linalg.norm(pa)) @ (pc / np.linalg.norm(pc)))
    assert cos_ab > 0.99
    assert abs(cos_ac) < 0.2
    # identique à la matrice exposée par vector_grid (même cache de projection)
    grid_a = emb.vector_grid("a", DIM_GRID)
    np.testing.assert_allclose(pa / np.linalg.norm(pa), grid_a, atol=1e-5)


def test_load_fichier_invalide_leve_valueerror(tmp_path):
    """Invalid .msee files (wrong magic, truncated) raise ValueError."""
    import pytest

    # Case (a): Wrong magic
    invalid_magic_path = tmp_path / "wrong_magic.msee"
    with open(invalid_magic_path, "wb") as f:
        f.write(b"XXXX" + b"\x00" * 20)

    with pytest.raises(ValueError, match="invalide"):
        Embeddings.load(invalid_magic_path)

    # Case (b): Truncated file (only first 6 bytes of a valid .msee)
    # Create a minimal valid file first, then truncate it
    src = _fake_vec_gz(tmp_path, [("a", 1)])
    full_path = tmp_path / "full.msee"
    prepare(src, full_path)

    truncated_path = tmp_path / "truncated.msee"
    with open(full_path, "rb") as f:
        data = f.read(6)  # Only first 6 bytes
    with open(truncated_path, "wb") as f:
        f.write(data)

    with pytest.raises(ValueError, match="tronquée|corrompue"):
        Embeddings.load(truncated_path)


# -- v1.5 : chargement paresseux (verify=False, chemin recherche) --------------------------


def test_load_verify_false_meme_resultat_que_verify_true_abtt_zero(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1), ("chien", 2), ("maison", 3)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    eager = Embeddings.load(out, verify=True)
    lazy = Embeddings.load(out, verify=False)
    assert eager.verified is True
    assert lazy.verified is False
    for w in ("chat", "chien", "maison"):
        np.testing.assert_array_equal(
            eager.vector_grid(w, DIM_GRID), lazy.vector_grid(w, DIM_GRID)
        )
    for w in ("chat", "chien", "maison"):
        np.testing.assert_array_equal(eager.raw_vector(w), lazy.raw_vector(w))


def test_load_verify_false_meme_resultat_que_verify_true_avec_abtt(tmp_path):
    rng = np.random.default_rng(42)
    base = rng.normal(size=300) * 5.0
    words = [f"mot{i}" for i in range(30)]
    lines = ["30 300"]
    for w in words:
        vec = base + rng.normal(size=300) * 0.3
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in vec))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "table.msee"
    prepare(src, out)

    eager = Embeddings.load(out, abtt=2, verify=True)
    lazy = Embeddings.load(out, abtt=2, verify=False)
    for w in words:
        np.testing.assert_array_equal(
            eager.vector_grid(w, DIM_GRID), lazy.vector_grid(w, DIM_GRID)
        )


def test_load_verify_false_ne_calcule_pas_le_sha(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    lazy = Embeddings.load(out, verify=False)
    assert lazy.sha == ""
    eager = Embeddings.load(out, verify=True)
    assert len(eager.sha) == 64


def test_load_verify_false_mat_est_un_memmap_quand_abtt_zero(tmp_path):
    src = _fake_vec_gz(tmp_path, [("chat", 1)])
    out = tmp_path / "table.msee"
    prepare(src, out)
    lazy = Embeddings.load(out, verify=False)
    assert isinstance(lazy.mat, np.memmap)


def test_load_verify_false_abtt_materialise_paresseusement_au_premier_appel(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.normal(size=300) * 5.0
    words = [f"mot{i}" for i in range(10)]
    lines = ["10 300"]
    for w in words:
        vec = base + rng.normal(size=300) * 0.3
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in vec))
    src = tmp_path / "mini.vec.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out = tmp_path / "table.msee"
    prepare(src, out)

    lazy = Embeddings.load(out, abtt=2, verify=False)
    assert lazy.mat is None  # rien de matérialisé tant qu'aucun accès réel n'a eu lieu
    assert lazy.vector_grid(words[0], DIM_GRID) is not None
    assert lazy.mat is not None
    assert not isinstance(
        lazy.mat, np.memmap
    )  # matrice réelle (abtt appliqué), pas un memmap


def test_load_verify_false_fichier_invalide_leve_valueerror(tmp_path):
    import pytest

    invalid_magic_path = tmp_path / "wrong_magic.msee"
    with open(invalid_magic_path, "wb") as f:
        f.write(b"XXXX" + b"\x00" * 20)
    with pytest.raises(ValueError, match="invalide"):
        Embeddings.load(invalid_magic_path, verify=False)

    src = _fake_vec_gz(tmp_path, [("a", 1)])
    full_path = tmp_path / "full.msee"
    prepare(src, full_path)
    truncated_path = tmp_path / "truncated.msee"
    with open(full_path, "rb") as f:
        data = f.read(6)
    with open(truncated_path, "wb") as f:
        f.write(data)
    with pytest.raises(ValueError, match="tronquée|corrompue"):
        Embeddings.load(truncated_path, verify=False)


# -- v1.6 §C : table pré-nettoyée (abtt appliqué à la préparation) -------------------------


def _base_noise_vec_gz(tmp_path, n=30, seed=42, name="mini.vec.gz"):
    """30 mots = base commune forte (même direction dominante) + bruit propre à chaque mot,
    comme test_abtt_reduit_la_composante_commune — matériel de test partagé pour les tests
    --abtt à la préparation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=300) * 5.0
    words = [f"mot{i}" for i in range(n)]
    lines = [f"{n} 300"]
    for w in words:
        vec = base + rng.normal(size=300) * 0.3
        lines.append(w + " " + " ".join(f"{v:.4f}" for v in vec))
    src = tmp_path / name
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return src, words


def test_prepare_abtt_bump_vmin_et_ecrit_abtt_applique(tmp_path):
    import struct

    from mosaic.embeddings import _HEADER

    src, _ = _base_noise_vec_gz(tmp_path)
    out = tmp_path / "table.msee"
    prepare(src, out, abtt=2)

    raw = out.read_bytes()
    _magic, vmaj, vmin, _n, _dim = _HEADER.unpack_from(raw, 0)
    assert vmaj == 1 and vmin == 1  # bump mineur, magic/majeur inchangés
    (abtt_applique,) = struct.unpack_from("<B", raw, _HEADER.size)
    assert abtt_applique == 2


def test_prepare_abtt_zero_pas_de_bump_de_version(tmp_path):
    from mosaic.embeddings import _HEADER

    src, _ = _base_noise_vec_gz(tmp_path)
    out = tmp_path / "table.msee"
    prepare(src, out, abtt=0)

    raw = out.read_bytes()
    _magic, vmaj, vmin, _n, _dim = _HEADER.unpack_from(raw, 0)
    assert vmaj == 1 and vmin == 0


def test_prepare_abtt_transforme_a_la_preparation_fast_path_identique_au_chemin_historique(
    tmp_path,
):
    """prepare(..., abtt=2) doit produire, une fois chargé avec le même abtt, exactement la
    même matrice que l'ancien chemin (table brute, transform appliqué à Embeddings.load)."""
    src, _ = _base_noise_vec_gz(tmp_path)
    out_pre = tmp_path / "pre.msee"
    out_brut = tmp_path / "brut.msee"
    prepare(src, out_pre, abtt=2)
    prepare(src, out_brut, abtt=0)

    fast = Embeddings.load(out_pre, abtt=2)  # déjà nettoyée à la préparation
    historique = Embeddings.load(
        out_brut, abtt=2
    )  # nettoyée au chargement (chemin v1.5)
    np.testing.assert_allclose(
        fast.mat.astype(np.float32), historique.mat.astype(np.float32), atol=1e-2
    )


def test_prepare_abtt_deterministe_octet_pour_octet(tmp_path):
    src, _ = _base_noise_vec_gz(tmp_path)
    out1 = tmp_path / "table1.msee"
    out2 = tmp_path / "table2.msee"
    prepare(src, out1, abtt=2)
    prepare(src, out2, abtt=2)
    assert out1.read_bytes() == out2.read_bytes()


def test_load_abtt_applique_egal_est_no_op_rapide_et_memmap_en_lazy(tmp_path):
    src, _ = _base_noise_vec_gz(tmp_path)
    out = tmp_path / "table.msee"
    prepare(src, out, abtt=2)

    lazy = Embeddings.load(out, abtt=2, verify=False)
    assert lazy.abtt == 2
    # jamais de PCA différée : le memmap est exposé tel quel (déjà nettoyé sur disque)
    assert isinstance(lazy.mat, np.memmap)

    eager = Embeddings.load(out, abtt=2, verify=True)
    assert eager.abtt == 2
    assert eager.mat is not None


def test_load_abtt_mismatch_leve_valueerror_eager_et_lazy(tmp_path):
    import pytest

    src, _ = _base_noise_vec_gz(tmp_path)
    out = tmp_path / "table.msee"
    prepare(src, out, abtt=2)

    with pytest.raises(ValueError, match="abtt"):
        Embeddings.load(out, abtt=3)
    with pytest.raises(ValueError, match="abtt"):
        Embeddings.load(out, abtt=0)
    with pytest.raises(ValueError, match="abtt"):
        Embeddings.load(out, abtt=3, verify=False)
    with pytest.raises(ValueError, match="abtt"):
        Embeddings.load(out, abtt=0, verify=False)


def test_load_abtt_zero_table_non_nettoyee_transforme_toujours_au_chargement(tmp_path):
    """Non-régression : une table vmin=0 (jamais nettoyée à la préparation, ancien format ou
    prepare(..., abtt=0)) continue de subir la transformation à Embeddings.load(abtt=N),
    exactement comme avant v1.6 §C."""
    src, _ = _base_noise_vec_gz(tmp_path)
    out = tmp_path / "table.msee"
    prepare(src, out)  # abtt=0 implicite

    def _mean_cosine(emb, words):
        vecs = emb.mat.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / norms
        cos = unit @ unit.T
        iu = np.triu_indices(len(words), k=1)
        return float(np.mean(cos[iu]))

    words = [f"mot{i}" for i in range(30)]
    emb0 = Embeddings.load(out, abtt=0)
    emb2 = Embeddings.load(out, abtt=2)
    assert emb0.abtt == 0 and emb2.abtt == 2
    assert _mean_cosine(emb2, words) < _mean_cosine(emb0, words) - 0.2
