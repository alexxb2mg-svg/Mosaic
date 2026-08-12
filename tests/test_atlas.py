"""Canal atlas (#367) : build/open/add, gardes, fusion à quatre canaux, stats.
Cf. spec docs/superpowers/specs/2026-08-12-atlas-canal-fusion-design.md."""

from pathlib import Path

import numpy as np
import pytest

from mosaic import rerank
from mosaic.index import Index

GRID = (32, 32, 3)


def _corpus(tmp_path: Path) -> Path:
    c = tmp_path / "corpus"
    c.mkdir(parents=True, exist_ok=True)
    (c / "elec.md").write_text(
        "pose interrupteur differentiel tableau protection disjoncteur circuit",
        encoding="utf-8",
    )
    (c / "carrelage.md").write_text(
        "achat carrelage gris cuisine colle joint pose sol", encoding="utf-8"
    )
    (c / "devis.md").write_text(
        "devis chantier peinture couloir escalier plafond", encoding="utf-8"
    )
    return c


def _fake_model(monkeypatch) -> None:
    """Vecteurs déterministes seedés par hash(texte) — même harnais que
    tests/test_index_rerank.py (pas de dépendance au vrai model2vec)."""

    class _FakeModel:
        def encode(self, texts):
            out = np.zeros((len(texts), rerank.DIM), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i] = np.random.default_rng(hash(t) & 0xFFFFFFFF).normal(
                    size=rerank.DIM
                )
            return out

    monkeypatch.setattr(rerank, "StaticModel", _FakeModel)
    monkeypatch.setattr(rerank, "_get_model", lambda: _FakeModel())


def _atlas(tmp_path: Path) -> Index:
    return Index.build(
        _corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True, atlas=True
    )


# -- gardes ---------------------------------------------------------------------------------


def test_atlas_exige_hybride(tmp_path):
    """Le canal n'est validé qu'en QUATUOR — refus net hors --hybride."""
    with pytest.raises(ValueError, match="hybride"):
        Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, atlas=True)


def test_atlas_refuse_grilles_typees(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    with pytest.raises(ValueError, match="typees"):
        Index.build(
            _corpus(tmp_path),
            tmp_path / "idx",
            grid=GRID,
            hybride=True,
            grilles_typees=True,
            atlas=True,
        )


# -- build / stockage -----------------------------------------------------------------------


def test_build_cree_les_fichiers_atlas(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = _atlas(tmp_path)
    assert (tmp_path / "idx" / "atlas.msat").is_file()
    assert (tmp_path / "idx" / "docs_atlas.msei").is_file()
    assert idx.atlas_positions is not None
    assert len(idx.atlas_positions) == len(idx.profiles.rows)
    assert idx.atlas_mat is not None and idx.atlas_mat.shape[0] == len(idx.ids)


def test_build_sans_atlas_ne_cree_rien(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    assert not (tmp_path / "idx" / "atlas.msat").exists()
    assert idx.atlas_positions is None


def test_determinisme_deux_builds_memes_fichiers(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    a = Index.build(
        _corpus(tmp_path), tmp_path / "a", grid=GRID, hybride=True, atlas=True
    )
    b = Index.build(
        _corpus(tmp_path), tmp_path / "b", grid=GRID, hybride=True, atlas=True
    )
    assert a.atlas_positions is not None and a.atlas_mat is not None
    assert b.atlas_positions is not None and b.atlas_mat is not None
    assert np.array_equal(a.atlas_positions, b.atlas_positions)
    assert np.array_equal(a.atlas_mat, b.atlas_mat)


# -- fusion ---------------------------------------------------------------------------------


def test_fusion_expose_quatre_rangs(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = _atlas(tmp_path)
    hits = idx.search("pose carrelage cuisine", k=3, fusion=True)
    assert hits, "des résultats sont attendus"
    assert "atlas" in hits[0]["rangs"], "le 4e canal doit être exposé"
    assert {"grille", "bm25", "embed", "atlas"} <= set(hits[0]["rangs"])


def test_fusion_trois_canaux_sans_atlas_intacte(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    idx = Index.build(_corpus(tmp_path), tmp_path / "idx", grid=GRID, hybride=True)
    hits = idx.search("pose carrelage", k=3, fusion=True)
    assert hits and "atlas" not in hits[0]["rangs"]


def test_requete_hors_vocabulaire_canal_ecarte(tmp_path, monkeypatch):
    """Requête sans AUCUN token mappé sur l'atlas : le canal est écarté sans erreur
    (même règle que les trois autres canaux aveugles)."""
    _fake_model(monkeypatch)
    idx = _atlas(tmp_path)
    hits = idx.search("zzz inexistant xxyy", k=3, fusion=True)
    assert all("atlas" not in h["rangs"] for h in hits)


def test_recherche_identique_apres_reouverture(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    avant = _atlas(tmp_path).search("pose interrupteur", k=3, fusion=True)
    apres = Index.open(tmp_path / "idx").search("pose interrupteur", k=3, fusion=True)
    assert avant == apres


# -- add ------------------------------------------------------------------------------------


def test_add_met_a_jour_les_cartes(tmp_path, monkeypatch):
    """Le doc ajouté combine des tokens CONNUS du build (sa carte atlas est donc non
    vide via le mapping figé) dans une combinaison unique : la fusion doit le classer
    premier, canal atlas actif compris."""
    _fake_model(monkeypatch)
    idx = _atlas(tmp_path)
    nouveau = tmp_path / "combi.md"
    nouveau.write_text(
        "carrelage interrupteur plafond carrelage interrupteur plafond",
        encoding="utf-8",
    )
    idx.add(nouveau)
    assert idx.atlas_mat is not None and idx.atlas_mat.shape[0] == len(idx.ids)
    hits = Index.open(tmp_path / "idx").search(
        "carrelage interrupteur plafond", k=3, fusion=True
    )
    assert hits and hits[0]["id"] == "combi.md"
    assert "atlas" in hits[0]["rangs"], "la carte du doc ajouté doit porter du signal"


# -- stats ----------------------------------------------------------------------------------


def test_stats_expose_atlas(tmp_path, monkeypatch):
    _fake_model(monkeypatch)
    s = _atlas(tmp_path).stats()
    assert s["atlas"]["cote"] == 64
    assert s["atlas"]["tokens_mappes"] > 0
    assert 0 < s["atlas"]["cellules_occupees"] <= 64 * 64
