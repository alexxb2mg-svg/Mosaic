"""Tests du repêcheur au niveau Index : build(rerank_vectors=...), search(rerank=...), add().

La plupart des tests utilisent un faux modèle (déterministe, seedé par hash du texte) pour
rester rapides et ne pas dépendre du vrai téléchargement/chargement de potion-128M — seul
test_encode_texts_modele_reel (tests/test_rerank.py) et les tests CLI dédiés couvrent
l'intégration réelle.
"""

from pathlib import Path

import numpy as np
import pytest

from mosaic import rerank
from mosaic.index import Index
from mosaic.store import load_rerank, load_vocab, save_rerank

GRID = (32, 32, 3)


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "elec1.md").write_text(
        "pose interrupteur différentiel salle de bain volume protection",
        encoding="utf-8",
    )
    (c / "elec2.md").write_text(
        "remplacement interrupteur disjoncteur différentiel tableau protection circuit",
        encoding="utf-8",
    )
    (c / "carrelage.md").write_text(
        "achat carrelage gris cuisine colle joint pose sol", encoding="utf-8"
    )
    return c


def _corpus5(tmp_path: Path) -> Path:
    c = tmp_path / "corpus5"
    c.mkdir(parents=True, exist_ok=True)
    docs = {
        "a.md": "interrupteur différentiel salle de bain protection",
        "b.md": "interrupteur disjoncteur tableau protection circuit",
        "c.md": "interrupteur simple va-et-vient chambre",
        "d.md": "carrelage gris cuisine colle joint",
        "e.md": "chauffe-eau ballon raccordement cuivre",
    }
    for name, text in docs.items():
        (c / name).write_text(text, encoding="utf-8")
    return c


def _fake_model(monkeypatch) -> None:
    """Vecteurs déterministes seedés par hash(texte) : identiques à chaque appel, pas de
    dépendance au vrai modèle model2vec."""

    class _FakeModel:
        def encode(self, texts):
            out = np.zeros((len(texts), rerank.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i] = np.random.default_rng(hash(t) & 0xFFFFFFFF).normal(
                    size=rerank.DIM
                )
            return out

    # available() (donc le garde de build rerank_vectors) doit passer sans le vrai
    # model2vec installé : on rend StaticModel non-None. Les tests qui veulent l'ABSENCE
    # au runtime reposent StaticModel=None eux-mêmes après le build.
    monkeypatch.setattr(rerank, "StaticModel", _FakeModel)
    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())


# -- build ----------------------------------------------------------------------------------


def test_build_rerank_vectors_cree_le_fichier_msrv(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    assert (tmp_path / "idx" / "rerank.msrv").is_file()
    assert idx.rerank_vecs is not None
    assert idx.rerank_vecs.shape == (3, rerank.DIM)
    assert idx.rerank_model == rerank.MODEL_NAME


def test_build_sans_rerank_vectors_ne_cree_rien(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert not (tmp_path / "idx" / "rerank.msrv").exists()
    assert idx.rerank_vecs is None
    assert idx.rerank_model is None


def test_build_rerank_vectors_meta_persiste_et_survit_a_open(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True)

    _, _, _, meta = load_vocab(tmp_path / "idx")
    assert meta["rerank_model"] == rerank.MODEL_NAME

    idx2 = Index.open(tmp_path / "idx")
    assert idx2.rerank_model == rerank.MODEL_NAME
    assert idx2.rerank_vecs.shape == (3, rerank.DIM)


def test_build_rerank_vectors_ordre_ids(tmp_path, monkeypatch):
    """Row i de rerank.msrv doit correspondre à idx.ids[i]."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    mat, _model = load_rerank(tmp_path / "idx")
    assert mat.shape[0] == len(idx.ids)


def test_build_rerank_vectors_identique_apres_reouverture(tmp_path, monkeypatch):
    """Preuve au niveau Index (pas seulement store.py) de la garantie de précision : les
    vecteurs en mémoire juste après build() sont bit-identiques à ceux relus par open()
    (arrondi float16 appliqué au même endroit des deux côtés, cf. _round_f16)."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    idx2 = Index.open(tmp_path / "idx")
    assert np.array_equal(idx.rerank_vecs, idx2.rerank_vecs)
    assert idx.rerank_vecs.dtype == idx2.rerank_vecs.dtype == np.float32


# -- stats : diagnostics -----------------------------------------------------------------------


def test_stats_rerank_model_present_avec_rerank_vectors(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    assert idx.stats()["rerank_model"] == rerank.MODEL_NAME


def test_stats_rerank_model_absent_sans_rerank_vectors(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    assert "rerank_model" not in idx.stats()


def test_build_rerank_vectors_sans_model2vec_leve_valueerror(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank, "StaticModel", None)
    with pytest.raises(ValueError):
        Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True)
    # dégradation propre : rien n'est créé sur disque avant l'échec
    assert not (tmp_path / "idx" / "rerank.msrv").exists()


def test_build_rerank_vectors_convertit_une_seule_fois(tmp_path, monkeypatch):
    """Le texte d'un fichier convertible (markitdown) ne doit être extrait qu'une fois
    (tokens ET vecteur rerank) — un compteur d'appels sur ingest.to_text le prouve."""
    from mosaic import ingest

    _fake_model(monkeypatch)
    calls = []

    def _fake_to_text(path, cache_dir=None, ocr=False):
        calls.append(path)
        return "cablage tableau protection"

    monkeypatch.setattr(ingest, "available", lambda: True)
    monkeypatch.setattr(ingest, "to_text", _fake_to_text)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "devis.pdf").write_bytes(
        b"peu importe, jamais lu reellement (to_text mocke)"
    )

    idx = Index.build(corpus, tmp_path / "idx", grid=GRID, rerank_vectors=True)

    assert len(calls) == 1  # une seule conversion pour tokens ET texte de rerank
    assert idx.stats()["docs"] == 1
    assert idx.rerank_vecs.shape == (1, rerank.DIM)


def test_build_rerank_vectors_byte_identique_reste_intact_sans_le_drapeau(
    tmp_path, monkeypatch
):
    """Non-régression : construire sans --rerank-vectors produit le même docs.msei/vocab.msev
    qu'avant cette fonctionnalité (le chemin par défaut n'est jamais touché)."""
    _fake_model(monkeypatch)
    Index.build(_corpus(tmp_path / "a"), tmp_path / "ia", grid=GRID)
    Index.build(
        _corpus(tmp_path / "b"), tmp_path / "ib", grid=GRID, rerank_vectors=False
    )
    assert (tmp_path / "ia" / "docs.msei").read_bytes() == (
        tmp_path / "ib" / "docs.msei"
    ).read_bytes()
    assert (tmp_path / "ia" / "vocab.msev").read_bytes() == (
        tmp_path / "ib" / "vocab.msev"
    ).read_bytes()


# -- open : cohérence ------------------------------------------------------------------------


def test_open_rerank_msrv_incoherent_leve_valueerror(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True)
    save_rerank(
        tmp_path / "idx", np.zeros((1, rerank.DIM), dtype=np.float32), rerank.MODEL_NAME
    )
    with pytest.raises(ValueError):
        Index.open(tmp_path / "idx")


# -- search -----------------------------------------------------------------------------------


def test_search_rerank_sans_msrv_leve_valueerror(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    with pytest.raises(ValueError):
        idx.search("interrupteur différentiel", k=2, rerank=True)


def test_search_rerank_sans_model2vec_leve_valueerror(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    monkeypatch.setattr(rerank, "StaticModel", None)
    with pytest.raises(ValueError):
        idx.search("interrupteur différentiel", k=2, rerank=True)


def test_search_rerank_ajoute_score_rerank(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    hits = idx.search("interrupteur différentiel protection", k=3, rerank=True)
    assert len(hits) == 3
    assert all("score_rerank" in h and "score" in h for h in hits)


def test_search_sans_rerank_pas_de_score_rerank(tmp_path, monkeypatch):
    """Non-régression : même avec rerank.msrv présent, sans --rerank la sortie est inchangée."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    hits = idx.search("interrupteur différentiel protection", k=3)
    assert all(set(h.keys()) == {"id", "score"} for h in hits)


def test_search_rerank_lambda_1_ordre_mosaic_pur(tmp_path, monkeypatch):
    """λ=1 => mélange == cos_mosaic pur => même ordre et mêmes valeurs que sans rerank."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    base = idx.search("interrupteur différentiel protection", k=3)
    reranked = idx.search(
        "interrupteur différentiel protection", k=3, rerank=True, rerank_lambda=1.0
    )
    assert [h["id"] for h in base] == [h["id"] for h in reranked]
    for b, r in zip(base, reranked, strict=True):
        assert abs(b["score"] - r["score_rerank"]) < 1e-5


def test_search_rerank_lambda_0_ordre_purement_m2v(tmp_path, monkeypatch):
    """λ=0 => score_rerank == cos_m2v exactement (canal mosaic ignoré dans le mélange)."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    query = "interrupteur différentiel protection"
    hits = idx.search(query, k=3, rerank=True, rerank_lambda=0.0)

    qvec = rerank.encode_query(query)
    mat, _ = load_rerank(tmp_path / "idx")
    mat = mat.astype(np.float32)
    expected = {idx.ids[i]: float(np.dot(mat[i], qvec)) for i in range(len(idx.ids))}
    for h in hits:
        assert abs(h["score_rerank"] - round(expected[h["id"]], 6)) < 1e-4


def test_search_rerank_depth_limite_le_reclassement(tmp_path, monkeypatch):
    """profondeur < nombre de docs : seuls les `depth` premiers (par score mosaic) sont
    re-classés (score_rerank), le reste garde l'ordre mosaic pur (pas de score_rerank)."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus5(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    hits = idx.search("interrupteur protection", k=5, rerank=True, rerank_depth=2)
    assert len(hits) == 5
    avec = [h for h in hits if "score_rerank" in h]
    sans = [h for h in hits if "score_rerank" not in h]
    assert len(avec) == 2
    assert len(sans) == 3


# -- add --------------------------------------------------------------------------------------


def test_add_met_a_jour_rerank_msrv_si_present(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx.add(extra)

    assert idx.rerank_vecs.shape[0] == 4
    mat, _ = load_rerank(tmp_path / "idx")
    assert mat.shape[0] == 4

    idx2 = Index.open(tmp_path / "idx")
    assert idx2.rerank_vecs.shape[0] == 4


def test_add_sans_rerank_msrv_reste_absent(tmp_path):
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID)
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )
    idx.add(extra)
    assert not (tmp_path / "idx" / "rerank.msrv").exists()
    assert idx.rerank_vecs is None


def test_add_rerank_present_sans_model2vec_leve_valueerror_et_ne_mute_rien(
    tmp_path, monkeypatch
):
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )

    monkeypatch.setattr(rerank, "StaticModel", None)
    with pytest.raises(ValueError):
        idx.add(extra)

    assert idx.stats()["docs"] == 3  # échec avant toute mutation (fail-fast)
    assert "plomberie.md" not in idx.ids


def test_add_rerank_encode_echoue_ne_laisse_pas_d_etat_partiel(tmp_path, monkeypatch):
    """model2vec DISPONIBLE (available() -> True) mais encode_texts() échoue quand même au
    runtime (poids corrompus, erreur du modèle...) : add() doit lever ET laisser self
    parfaitement inchangé — ids/mat/rerank_vecs ne doivent jamais désynchroniser d'un cran."""
    _fake_model(monkeypatch)
    idx = Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, rerank_vectors=True
    )
    extra = tmp_path / "plomberie.md"
    extra.write_text(
        "chauffe-eau ballon eau chaude raccordement cuivre", encoding="utf-8"
    )

    n_ids_avant = len(idx.ids)
    mat_shape_avant = idx.mat.shape
    norms_len_avant = len(idx.norms)
    rerank_shape_avant = idx.rerank_vecs.shape
    vocab_avant = dict(idx.profiles.rows)

    def _boom(texts):
        raise RuntimeError(
            "échec runtime simulé (modèle disponible mais encodage en échec)"
        )

    monkeypatch.setattr(rerank, "encode_texts", _boom)

    with pytest.raises(RuntimeError):
        idx.add(extra)

    assert len(idx.ids) == n_ids_avant
    assert "plomberie.md" not in idx.ids
    assert idx.mat.shape == mat_shape_avant
    assert len(idx.norms) == norms_len_avant
    assert idx.rerank_vecs.shape == rerank_shape_avant
    assert (
        idx.profiles.rows == vocab_avant
    )  # le vocabulaire n'a pas non plus appris "ballon" etc.
