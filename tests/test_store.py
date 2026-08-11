import json
import struct

import numpy as np

from mosaic.index import Index
from mosaic.profiles import Profiles
from mosaic.signatures import signature
from mosaic.store import (
    load_docs,
    load_rerank,
    load_vocab,
    save_docs,
    save_rerank,
    save_vocab,
)

GRID = (32, 32, 3)
DIM = 3072


def test_docs_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    mat = rng.integers(-127, 128, size=(4, DIM), dtype=np.int8)
    norms = np.linalg.norm(mat.astype(np.float32), axis=1)
    ids = ["a.md", "différentiel.md", "sous/dossier/électricité.md", "d.md"]
    save_docs(tmp_path, mat, norms, ids, GRID)
    mat2, norms2, ids2, grid2 = load_docs(tmp_path)
    assert np.array_equal(mat, mat2)
    assert np.allclose(norms, norms2)
    assert ids2 == ids and grid2 == GRID


def test_vocab_round_trip(tmp_path):
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    lex = {"breaker": "disjoncteur"}
    save_vocab(tmp_path, p, colloc, GRID, lex)
    p2, colloc2, lex2, meta2 = load_vocab(tmp_path)
    assert colloc2 == colloc
    assert lex2 == lex
    assert p2.n_docs == p.n_docs and p2.df == p.df and p2.counts == p.counts
    assert p2.rows == p.rows
    n = len(p.rows)
    assert np.array_equal(p2.acc[:n], p.acc[:n])
    assert p2.acc.dtype == np.float32
    v1, v2 = p.word_vector("interrupteur"), p2.word_vector("interrupteur")
    assert np.allclose(v1, v2)


def test_magic_invalide(tmp_path):
    (tmp_path / "docs.msei").write_bytes(b"XXXX" + b"\x00" * 32)
    try:
        load_docs(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass

    (tmp_path / "vocab.msev").write_bytes(b"YYYY" + b"\x00" * 32)
    try:
        load_vocab(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


def test_fichier_tronque_leve_valueerror(tmp_path):
    (tmp_path / "docs.msei").write_bytes(b"\x00" * 10)
    try:
        load_docs(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass

    (tmp_path / "vocab.msev").write_bytes(b"\x00" * 10)
    try:
        load_vocab(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


def test_vocab_deterministe_entre_deux_ecritures(tmp_path):
    def _make() -> Profiles:
        p = Profiles(DIM)
        p.learn(["interrupteur", "différentiel", "salle", "bain"])
        p.learn(["interrupteur", "va-et-vient"])
        p.finalize("brut")
        return p

    colloc = {("interrupteur", "différentiel")}
    lex = {"breaker": "disjoncteur"}
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    save_vocab(dir_a, _make(), colloc, GRID, lex)
    save_vocab(dir_b, _make(), colloc, GRID, lex)
    assert (dir_a / "vocab.msev").read_bytes() == (dir_b / "vocab.msev").read_bytes()


def test_msev_v11_round_trip_et_add(tmp_path):
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    lex = {"breaker": "disjoncteur"}
    save_vocab(tmp_path, p, colloc, GRID, lex, extra_meta={"profile_weighting": "brut"})

    p2, colloc2, lex2, meta2 = load_vocab(tmp_path)
    assert p2.pair_counts == p.pair_counts
    assert p2.marginals == p.marginals
    assert p2.total_mass == p.total_mass
    assert meta2.get("legacy") is False
    assert meta2.get("profile_weighting") == "brut"

    # add() fonctionne : ré-apprentissage à partir de l'état rechargé, puis re-finalize
    p2.learn(["disjoncteur", "tableau"])
    p2.finalize("brut")
    assert "disjoncteur" in p2.rows and "tableau" in p2.rows
    assert p2.acc.shape[0] == len(p2.rows)


def test_msev_v10_legacy_charge_mais_add_refuse(tmp_path):
    # Fabrique à la main une paire docs.msei/vocab.msev au format v1.0 (avant cette
    # tâche : pas de bloc épars, acc stocké en int32).
    ids = ["a.md"]
    mat = np.zeros((1, DIM), dtype=np.int8)
    norms = np.array([0.0], dtype=np.float32)
    save_docs(tmp_path, mat, norms, ids, GRID)

    order = ["cable", "cuivre"]
    acc = np.zeros((2, DIM), dtype=np.int32)
    acc[0] = 5 * signature("cuivre", DIM)
    acc[1] = 5 * signature("cable", DIM)
    meta = {
        "order": order,
        "counts": {"cable": 1, "cuivre": 1},
        "df": {"cable": 1, "cuivre": 1},
        "n_docs": 1,
        "colloc": [],
        "lexicon": {},
    }
    blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = struct.Struct("<4sBBHHBBI").pack(
        b"MSEV", 1, 0, GRID[0], GRID[1], GRID[2], 0, 2
    )
    with open(tmp_path / "vocab.msev", "wb") as f:
        f.write(header)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(acc.tobytes())

    p, colloc, lex, meta2 = load_vocab(tmp_path)
    assert meta2["legacy"] is True
    assert p.acc.dtype == np.float32
    assert np.array_equal(p.acc[0], (5 * signature("cuivre", DIM)).astype(np.float32))
    assert p.pair_counts == {} and p.marginals == {} and p.total_mass == 0.0

    idx = Index.open(tmp_path)
    try:
        idx.add(
            tmp_path / "nexiste_pas.md"
        )  # ne doit jamais aller jusqu'à la lecture du fichier
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


def test_msev_v11_tronque_dans_le_bloc_eparse_leve_valueerror(tmp_path):
    """Un vocab.msev v1.1 valide, mais tronqué EN PLEIN bloc épars (après acc, avant la fin
    de total_mass) doit lever ValueError — jamais laisser fuiter struct.error brut."""
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    lex = {"breaker": "disjoncteur"}
    save_vocab(tmp_path, p, colloc, GRID, lex)

    chemin = tmp_path / "vocab.msev"
    data = bytearray(chemin.read_bytes())
    assert len(data) > 4  # sanity : fichier valide non vide
    del data[
        -4:
    ]  # coupe les 4 derniers octets : struct.unpack_from("<d", ...) pour total_mass échoue
    chemin.write_bytes(bytes(data))

    try:
        load_vocab(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError as exc:
        assert "tronqué" in str(exc) or "corrompu" in str(exc)
    except struct.error:
        raise AssertionError(
            "struct.error a fuité au lieu d'être converti en ValueError"
        ) from None


def test_msev_vmin_non_supporte_leve_valueerror(tmp_path):
    order = ["cable"]
    meta = {
        "order": order,
        "counts": {"cable": 1},
        "df": {"cable": 1},
        "n_docs": 1,
        "colloc": [],
        "lexicon": {},
    }
    blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = struct.Struct("<4sBBHHBBI").pack(
        b"MSEV", 1, 2, GRID[0], GRID[1], GRID[2], 0, 1
    )
    with open(tmp_path / "vocab.msev", "wb") as f:
        f.write(header)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(np.zeros((1, DIM), dtype=np.float32).tobytes())

    try:
        load_vocab(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


# -- rerank.msrv (v1.4 le repêcheur) ------------------------------------------------------


def _unit_rows(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(n, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def test_rerank_absent_retourne_none(tmp_path):
    assert load_rerank(tmp_path) is None


def test_rerank_round_trip(tmp_path):
    mat = _unit_rows(5, 256, seed=3)
    save_rerank(tmp_path, mat, "minishlab/potion-multilingual-128M")
    mat2, model2 = load_rerank(tmp_path)
    assert model2 == "minishlab/potion-multilingual-128M"
    assert mat2.shape == (5, 256)
    assert mat2.dtype == np.float16
    assert np.allclose(mat2.astype(np.float32), mat, atol=2e-3)  # arrondi float16


def test_rerank_deterministe_entre_deux_ecritures(tmp_path):
    mat = _unit_rows(4, 256, seed=7)
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    save_rerank(dir_a, mat, "modele-x")
    save_rerank(dir_b, mat, "modele-x")
    assert (dir_a / "rerank.msrv").read_bytes() == (dir_b / "rerank.msrv").read_bytes()


def test_rerank_magic_invalide(tmp_path):
    (tmp_path / "rerank.msrv").write_bytes(b"XXXX" + b"\x00" * 32)
    try:
        load_rerank(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


def test_rerank_header_tronque_leve_valueerror(tmp_path):
    (tmp_path / "rerank.msrv").write_bytes(b"\x00" * 5)
    try:
        load_rerank(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


def test_rerank_matrice_tronquee_leve_valueerror(tmp_path):
    mat = _unit_rows(3, 256, seed=9)
    save_rerank(tmp_path, mat, "modele")
    chemin = tmp_path / "rerank.msrv"
    data = bytearray(chemin.read_bytes())
    assert len(data) > 4
    del data[-4:]  # coupe la fin de la matrice float16
    chemin.write_bytes(bytes(data))
    try:
        load_rerank(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass


# -- vocab.msev v1.1 en chargement paresseux (v1.5, Index.open côté recherche) -----------------


def test_load_vocab_lazy_acc_est_un_memmap(tmp_path):
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    lex = {"breaker": "disjoncteur"}
    save_vocab(tmp_path, p, colloc, GRID, lex)

    p2, colloc2, lex2, meta2 = load_vocab(tmp_path, lazy=True)
    assert isinstance(p2.acc, np.memmap)
    assert p2.acc.shape == p.acc.shape
    assert colloc2 == colloc and lex2 == lex
    assert p2.rows == p.rows and p2.df == p.df and p2.n_docs == p.n_docs
    # Bloc épars non analysé tant que rien ne le déclenche.
    assert p2.pair_counts == {} and p2.marginals == {} and p2.total_mass == 0.0
    assert p2._pending_sparse is not None


def test_load_vocab_lazy_valeurs_identiques_a_eager(tmp_path):
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("ppmi")
    colloc = {("interrupteur", "différentiel")}
    save_vocab(tmp_path, p, colloc, GRID, {})

    p_eager, _, _, _ = load_vocab(tmp_path, lazy=False)
    p_lazy, _, _, _ = load_vocab(tmp_path, lazy=True)
    assert np.array_equal(np.asarray(p_lazy.acc), p_eager.acc)
    v_eager = p_eager.word_vector("interrupteur")
    v_lazy = p_lazy.word_vector("interrupteur")
    assert np.array_equal(v_eager, v_lazy)


def test_load_vocab_lazy_materialise_le_bloc_eparse_sans_mutation(tmp_path):
    """_materialize_sparse() seul (sans apprendre de nouveau document) restitue exactement
    le pair_counts/marginals/total_mass d'origine — le chargeur différé n'invente rien."""
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    save_vocab(tmp_path, p, colloc, GRID, {})

    p2, _, _, _ = load_vocab(tmp_path, lazy=True)
    p2._materialize_sparse()
    assert p2._pending_sparse is None
    assert p2.pair_counts == p.pair_counts
    assert p2.marginals == p.marginals
    assert p2.total_mass == p.total_mass


def test_load_vocab_lazy_puis_learn_declenche_la_materialisation(tmp_path):
    """learn() (donc add() côté Index) déclenche la matérialisation du bloc épars AVANT
    d'accumuler le nouveau document — les anciens comptages restent la base, le nouveau
    document s'ajoute par-dessus (additivité normale de learn())."""
    p = Profiles(DIM)
    p.learn(["interrupteur", "différentiel", "salle", "bain"])
    p.learn(["interrupteur", "va-et-vient"])
    p.finalize("brut")
    colloc = {("interrupteur", "différentiel")}
    save_vocab(tmp_path, p, colloc, GRID, {})

    p2, _, _, _ = load_vocab(tmp_path, lazy=True)
    p2.learn(["tableau", "disjoncteur"])  # déclenche _materialize_sparse() puis apprend
    assert p2._pending_sparse is None
    # Les paires d'origine sont toutes encore présentes (additivité).
    assert set(p.pair_counts).issubset(p2.pair_counts)
    for key, w in p.pair_counts.items():
        assert p2.pair_counts[key] == w
    p2.finalize("brut")
    assert "tableau" in p2.rows and "disjoncteur" in p2.rows
    assert isinstance(p2.acc, np.ndarray) and not isinstance(p2.acc, np.memmap)


def test_load_vocab_lazy_legacy_v10_meme_comportement_qu_eager(tmp_path):
    ids = ["a.md"]
    mat = np.zeros((1, DIM), dtype=np.int8)
    norms = np.array([0.0], dtype=np.float32)
    save_docs(tmp_path, mat, norms, ids, GRID)

    order = ["cable", "cuivre"]
    acc = np.zeros((2, DIM), dtype=np.int32)
    acc[0] = 5 * signature("cuivre", DIM)
    acc[1] = 5 * signature("cable", DIM)
    meta = {
        "order": order,
        "counts": {"cable": 1, "cuivre": 1},
        "df": {"cable": 1, "cuivre": 1},
        "n_docs": 1,
        "colloc": [],
        "lexicon": {},
    }
    blob = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    header = struct.Struct("<4sBBHHBBI").pack(
        b"MSEV", 1, 0, GRID[0], GRID[1], GRID[2], 0, 2
    )
    with open(tmp_path / "vocab.msev", "wb") as f:
        f.write(header)
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(acc.tobytes())

    p, colloc, lex, meta2 = load_vocab(tmp_path, lazy=True)
    assert meta2["legacy"] is True
    assert isinstance(p.acc, np.ndarray) and not isinstance(p.acc, np.memmap)
    assert p.acc.dtype == np.float32
    assert np.array_equal(p.acc[0], (5 * signature("cuivre", DIM)).astype(np.float32))


def test_rerank_vmin_non_supporte_leve_valueerror(tmp_path):
    mat = _unit_rows(2, 256, seed=11)
    save_rerank(tmp_path, mat, "modele")
    chemin = tmp_path / "rerank.msrv"
    data = bytearray(chemin.read_bytes())
    # _HEADER = "<4sBBHHBBI" : magic(4s)=0-3, vmaj(B)=4, vmin(B)=5 — même octet que pour
    # vocab.msev (cf. test_msev_vmin_non_supporte_leve_valueerror).
    data[5] = 99
    chemin.write_bytes(bytes(data))
    try:
        load_rerank(tmp_path)
        raise AssertionError("aurait dû lever ValueError")
    except ValueError:
        pass
